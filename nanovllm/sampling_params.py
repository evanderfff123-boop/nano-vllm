from dataclasses import dataclass


# @dataclass是一个装饰器，自动写代码中繁琐但必要的方法：__init__、__repr__、__eq__等
# slots=True: 这是一个性能优化选项。它会限制类属性的动态添加，减少内存占用并稍微提升访问速度。
@dataclass(slots=True)
class SamplingParams:
    # 定义类属性及其默认值
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    # __post_init__ 是 dataclass 自动调用的一个特殊方法
    # 它在 __init__ 执行完毕后被调用，用于进行参数合法性校验
    def __post_init__(self):
        # 强制要求 temperature 必须大于一个极小值
        # 这里的 1e-10 实际上是 0.0000000001
        # 意思是：不允许使用温度为 0 或接近 0 的“贪婪采样”模式
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
