import torch
from torch import nn


class Sampler(nn.Module):
    """
    采样器类：负责根据 logits 生成最终的 token ID。
    使用了 Gumbel-Max 采样技巧，在 GPU 上具有极高的并行效率。
    """
    @torch.compile # 使用 Torch 编译加速计算，减少 python 开销
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        参数:
            logits: 模型最后一层的输出，形状为 [batch_size, vocab_size]
            temperatures: 采样温度值，形状为 [batch_size, 1] 或 [batch_size]
        """
        # 1. 温度缩放：调整分布的平滑度
        # 结果越小，分布越尖锐，模型输出越确定；越大则越具有随机性
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # 2. 转换为概率分布
        probs = torch.softmax(logits, dim=-1)
        # 3. Gumbel-Max 采样技巧
        # 这里的核心数学原理是：从分类分布 P 中采样，等价于计算 argmax(log(P) + Gumbel_noise)
        # 代码通过除以指数分布的噪声来实现采样，比传统的 torch.multinomial 更高效
        # exponential_(1) 生成服从指数分布的随机数
        # 加上 clamp_min 防止噪声过小导致除以零错误
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
