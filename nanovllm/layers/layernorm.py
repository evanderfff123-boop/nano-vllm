import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size)) # 可学习的缩放参数

    @torch.compile # 使用 PyTorch 2.0 编译优化，将下述操作融合为一个 CUDA Kernel
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float() # 转为 fp32 以保证平方和均值的计算精度
        var = x.pow(2).mean(dim=-1, keepdim=True) # 计算均方值
        x.mul_(torch.rsqrt(var + self.eps)) # 归一化：x / sqrt(var + eps)
        x = x.to(orig_dtype).mul_(self.weight) # 转回原精度并应用缩放权重
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        融合算子：将残差相加与 RMSNorm 合并
        """
        orig_dtype = x.dtype
        # 将当前层输出 x 与残差 residual 相加
        x = x.float().add_(residual.float()) 
        residual = x.to(orig_dtype) # 更新后的残差传给下一层
        # 对相加后的结果进行 RMSNorm
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            # 推理框架中常用的路径，减少显存读写，提升速度
            return self.add_rms_forward(x, residual)
