from __future__ import annotations

import math
from dataclasses import field
from math import gcd
from typing import List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


# helpers


def exists(val):
    return val is not None


def lcm(*numbers: int) -> int:
    return int(math.prod(numbers) // math.prod([gcd(x, y) for x, y in zip(numbers, numbers)])) if numbers else 1


def lcm_list(numbers: Sequence[int]) -> int:
    result = 1
    for number in numbers:
        result = int((result * number) // gcd(result, number))
    return result


def masked_mean(tensor: jnp.ndarray, mask: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    # Broadcast mask to tensor rank
    expand_dims = tensor.ndim - mask.ndim
    for _ in range(expand_dims):
        mask = mask[..., None]

    mask = mask.astype(tensor.dtype)
    tensor_masked = tensor * mask

    total_el = jnp.sum(mask, axis=axis)
    denom = jnp.clip(total_el, a_min=1.0)
    mean = jnp.sum(tensor_masked, axis=axis) / denom
    # when total_el == 0, set mean to 0
    mean = jnp.where(total_el == 0.0, jnp.zeros_like(mean), mean)
    return mean


def next_divisible_length(seqlen: int, multiple: int) -> int:
    if multiple <= 0:
        return seqlen
    return int(math.ceil(seqlen / multiple) * multiple)


def pad_to_multiple(
    tensor: jnp.ndarray,
    multiple: int,
    *,
    seq_axis: int,
    value: float | int = 0,
) -> jnp.ndarray:
    if multiple <= 0:
        return tensor
    seqlen = tensor.shape[seq_axis]
    length = next_divisible_length(seqlen, multiple)
    if length == seqlen:
        return tensor
    remainder = length - seqlen

    pad_config = [(0, 0)] * tensor.ndim
    pad_config[seq_axis] = (0, remainder)
    return jnp.pad(tensor, pad_config, constant_values=value)


class DepthwiseConv1d(nn.Module):
    dim_in: int
    dim_out: int
    kernel_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (batch, length, channels)
        # depthwise conv
        depthwise = nn.Conv(
            features=self.dim_in,
            kernel_size=(self.kernel_size,),
            feature_group_count=self.dim_in,
            use_bias=True,
            padding="VALID",
            name="depthwise",
        )
        pointwise = nn.Conv(
            features=self.dim_out,
            kernel_size=(1,),
            use_bias=True,
            padding="VALID",
            name="pointwise",
        )
        x = depthwise(x)
        x = pointwise(x)
        return x


class GBST(nn.Module):
    # Configuration
    num_tokens: int
    dim: int
    max_block_size: Optional[int] = None
    blocks: Optional[Tuple[int | Tuple[int, int], ...]] = None
    downsample_factor: int = 4
    score_consensus_attn: bool = True

    # Derived/initialized in setup
    _blocks: Tuple[Tuple[int, int], ...] = field(init=False, default=())
    _block_pad_multiple: int = field(init=False, default=1)

    def setup(self):
        assert exists(self.max_block_size) ^ exists(self.blocks), (
            "either max_block_size or blocks are given on initialization"
        )

        if exists(self.blocks):
            assert isinstance(self.blocks, tuple), "blocks must be a tuple of block sizes"
            normalized: List[Tuple[int, int]] = []
            for element in self.blocks:
                if isinstance(element, tuple):
                    block_size, offset = element
                else:
                    block_size, offset = int(element), 0
                normalized.append((block_size, offset))
            assert all([(offset < block_size) for block_size, offset in normalized]), (
                "offset must be always smaller than the block size"
            )
            max_block_size = max([b for b, _ in normalized])
            self._blocks = tuple(normalized)
            self._max_block_size = max_block_size
        else:
            assert exists(self.max_block_size)
            self._blocks = tuple((el, 0) for el in range(1, int(self.max_block_size) + 1))
            self._max_block_size = int(self.max_block_size)

        assert self.downsample_factor <= self._max_block_size, (
            "final downsample factor should be less than the maximum block size"
        )

        self._block_pad_multiple = lcm_list([block_size for block_size, _ in self._blocks])

        # Modules
        self.token_emb = nn.Embed(num_embeddings=self.num_tokens, features=self.dim, name="token_emb")
        self.pos_conv = DepthwiseConv1d(self.dim, self.dim, kernel_size=self._max_block_size)
        self.score_dense = nn.Dense(features=1, name="score_dense")

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        # x: (batch, length) int token ids
        # mask: (batch, length) bool
        batch_size, seqlen = x.shape[0], x.shape[1]
        block_mult = self._block_pad_multiple
        ds_factor = self.downsample_factor
        truncate_len = next_divisible_length(seqlen, ds_factor)

        # token embeddings
        x = self.token_emb(x)

        # positional conv: manually pad right by max_block_size - 1, then VALID depthwise conv
        right_pad = self._max_block_size - 1
        if right_pad > 0:
            x = jnp.pad(x, ((0, 0), (0, right_pad), (0, 0)))
        x = self.pos_conv(x)

        # pad sequence to be divisible by lcm of block sizes
        x = pad_to_multiple(x, block_mult, seq_axis=1, value=0.0)
        if exists(mask):
            mask = pad_to_multiple(mask.astype(bool), block_mult, seq_axis=1, value=False)

        # compute block representations (mean pooled)
        block_repr_list: List[jnp.ndarray] = []
        block_masks_list: List[jnp.ndarray] = []

        for block_size, offset in self._blocks:
            block_x = x
            block_mask = mask if exists(mask) else None

            # pad for offsets if needed
            if offset > 0:
                left_offset, right_offset = (block_size - offset), offset
                block_x = jnp.pad(block_x, ((0, 0), (left_offset, right_offset), (0, 0)), constant_values=0.0)
                if exists(block_mask):
                    block_mask = jnp.pad(
                        block_mask, ((0, 0), (left_offset, right_offset)), constant_values=False
                    )

            # group into blocks of length block_size
            total_len = block_x.shape[1]
            num_blocks = total_len // block_size
            # reshape: (b, n_blocks, block_size, d)
            block_groups = block_x.reshape(batch_size, num_blocks, block_size, self.dim)

            if exists(block_mask):
                mask_groups = block_mask.reshape(batch_size, num_blocks, block_size)
                block_repr = masked_mean(block_groups, mask_groups, axis=2)
            else:
                block_repr = jnp.mean(block_groups, axis=2)

            # repeat each block representation block_size times along sequence axis
            block_repr_repeated = jnp.repeat(block_repr, repeats=block_size, axis=1)

            # remove offset padding if it was added
            if offset > 0:
                block_repr_repeated = block_repr_repeated[:, left_offset : -right_offset]

            block_repr_list.append(block_repr_repeated)

            if exists(block_mask):
                block_mask_blocks = jnp.any(mask_groups, axis=-1)
                block_mask_repeated = jnp.repeat(block_mask_blocks, repeats=block_size, axis=1)
                if offset > 0:
                    block_mask_repeated = block_mask_repeated[:, left_offset : -right_offset]
                block_masks_list.append(block_mask_repeated)

        # stack along block-size dimension -> (b, n, num_block_configs, d)
        block_reprs = jnp.stack(block_repr_list, axis=2)

        # scores over block-size dimension
        scores = self.score_dense(block_reprs).squeeze(-1)  # (b, n, m)
        if exists(mask):
            block_masks = jnp.stack(block_masks_list, axis=2)  # (b, n, m)
            max_neg = -jnp.finfo(scores.dtype).max
            scores = jnp.where(block_masks, scores, max_neg)

        scores = jax.nn.softmax(scores, axis=2)
        scores = jnp.nan_to_num(scores, nan=0.0)

        # optional consensus attention
        if self.score_consensus_attn:
            # (b, n, n)
            score_sim = jnp.einsum("b i m, b j m -> b i j", scores, scores)
            if exists(mask):
                cross_mask = (mask[:, :, None] & mask[:, None, :]).astype(bool)
                max_neg = -jnp.finfo(score_sim.dtype).max
                score_sim = jnp.where(cross_mask, score_sim, max_neg)
            score_attn = jax.nn.softmax(score_sim, axis=-1)
            score_attn = jnp.nan_to_num(score_attn, nan=0.0)
            scores = jnp.einsum("b i j, b j m -> b i m", score_attn, scores)

        # weighted sum over block representations
        scores_expanded = scores[..., None]  # (b, n, m, 1)
        x = jnp.sum(block_reprs * scores_expanded, axis=2)  # (b, n, d)

        # truncate to length divisible by downsample factor (relative to original length)
        x = x[:, :truncate_len]
        if exists(mask):
            mask = mask[:, :truncate_len]

        # final mean pooling downsample
        # reshape: (b, n/df, df, d)
        df = ds_factor
        new_len = x.shape[1] // df
        x_blocks = x.reshape(batch_size, new_len, df, self.dim)
        if exists(mask):
            mask_blocks = mask.reshape(batch_size, new_len, df)
            x = masked_mean(x_blocks, mask_blocks, axis=2)
            mask = jnp.any(mask_blocks, axis=-1)
        else:
            x = jnp.mean(x_blocks, axis=2)

        return x, mask


