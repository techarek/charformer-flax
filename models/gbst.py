from flax import linen as nn

class GBST(nn.Module):
    radius: int = 128
    num_heads: int = 8
    num_memory_heads: int = 0
    key_value_size: int = 128
    shared_kv: bool = False
    dropout_rate: float = 0.0
    attention_kwargs: dict = None
    downsample_query: int = 2
    low_rank_features: int = 32
    project_kv: bool = True
    use_ffn: bool = True
    num_memory_slots: int = 0
    structured: bool = False
    pre_attention: bool = False
    local_gate: bool = False
    norm: bool = False
    pos_att: bool = False
    conv_type: str = None
    query_func: str = "linear"
    pool_func: str = "max"
    local_attention: bool = False
    use_offsets: bool = False
    consider_chars_as_blocks: bool = False
    use_block_pos_embedding: bool = False
    canine_mode: bool = False
    filter_size: int = 5
    block_mixing_mode: str = None
    rank_activation: str = "softmax"
    gbst_pool: str = "mean"

    def __call__(self, x):
        return x

if __name__ == "__main__":
    pass