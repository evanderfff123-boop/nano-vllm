import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

# --- 1. Triton 内核：高效存储 KV Cache ---
@triton.jit
def store_kvcache_kernel(
    key_ptr,  # 当前计算出的新 Key 的指针
    key_stride,  # Key 的步长（行间距）
    value_ptr,  # 当前计算出的新 Value 的指针
    value_stride,  # Value 的步长
    k_cache_ptr,  # 预分配好的全局 Key Cache 池指针
    v_cache_ptr,  # 预分配好的全局 Value Cache 池指针
    slot_mapping_ptr,  # 槽位映射：记录当前 Token 应该存放在 Cache 池的哪个绝对位置
    D: tl.constexpr,  # 每个 Token 对应的所有 Head 的总维度 (num_heads * head_dim)
):
    # 获取当前处理的 Token 索引（并行处理每个 Token）
    idx = tl.program_id(0)
    # 读取该 Token 对应的缓存槽位编号
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return # 如果槽位无效则跳过（例如填充 Token）

    # 计算该 Token 在输入张量中的偏移量
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    # 加载当前 Token 的 KV 数据
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # 计算在全局 Cache 池中的目标偏移量（Slot 决定了存放的位置）
    cache_offsets = slot * D + tl.arange(0, D)

    # 将数据存入全局池
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    封装 Triton 内核的函数。
    将计算出的 K, V 映射到非连续的显存块中。
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    # 简单的格式校验
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N

    # 启动 Triton Kernel。并行规模为 N (Token 数量)
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

# --- 2. Attention 模块 ---
class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # 这两个会被 ModelRunner 初始化时动态绑定到预分配的大显存块
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # 获取当前推理请求的元数据（是否为 Prefill，Block Table 等）
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 1. 存储 KV 到缓存池
        # 如果当前有分配好的缓存空间，则使用 Triton 内核把刚算出来的 K, V 塞进去
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # 2. 计算注意力 (分阶段处理)
        if context.is_prefill:
            # --- 预填充阶段 (Prefill) ---
            # 如果存在 prefix cache（前缀缓存命中），则直接使用缓存中的 K, V
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache

            # 使用 flash_attn 的变长序列接口（处理 Batch 中长度不一的情况）
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:
            # decode
            # --- 解码阶段 (Decode) ---
            # 每次只生成一个 Token，使用专门针对 KV Cache 优化的 FlashAttention 接口
            # q.unsqueeze(1) 将形状从 [BS, D] 转为 [BS, 1, D]
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
