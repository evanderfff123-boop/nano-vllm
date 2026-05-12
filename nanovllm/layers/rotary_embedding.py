from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """对输入张量 x 应用旋转位置编码"""
    # 将输入特征维度对半分，进行复数旋转变换
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # RoPE 公式: [cos -sin; sin cos] * [x1; x2]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    # 拼接并恢复原始精度
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
         # 计算旋转频率，频率随维度递减
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)

        # 外积生成频率表
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()

        # 缓存 cos 和 sin，以便快速检索
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile # 使用 Torch 编译优化计算
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 从缓存中根据位置索引提取对应 cos/sin
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        # 对 Query 和 Key 分别应用旋转
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1) # 单例模式，防止重复初始化相同的 RoPE 对象
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    """获取/初始化旋转位置编码实例"""
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
