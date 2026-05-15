import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # 1. 定义并解析模型所在的本地路径
    # 1. expanduser 是干嘛的？
    # 作用：os.path.expanduser("~/huggingface/...") 会把 ~ 替换为你的真实路径
    # （如 /home/user/huggingface/...）。如果没有这一步，
    # 程序会去寻找根目录下名为 ~ 的文件夹，从而导致找不到文件。
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # 2. 加载与模型匹配的分词器 (Tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(path)
    # 3. 初始化推理引擎
    # enforce_eager=True: 强制使用即时执行模式（便于调试）
    # 2. 即时执行模式（Eager Mode）是什么意思？
    # 
    # tensor_parallel_size=1: 设置 GPU 并行数量为 1
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # 4. 配置生成参数
    # temperature: 控制随机性，0.6 比较平衡
    # max_tokens: 限制单次生成的最大长度为 256
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

    # 定义待测试的问题列表
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]

    # 5. 对 Prompt 进行格式化 (Chat Template)
    # 将原始文本转换为模型指定的对话格式 (如添加 <|im_start|>user 等标记)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], # 构造标准的对话字典
            tokenize=False,                        # 不直接转为 Token ID，保留字符串
            add_generation_prompt=True,            # 自动添加 AI 回答的起始标记
        )
        for prompt in prompts
    ]

    # 6. 执行推理，传入提示词列表和参数
    outputs = llm.generate(prompts, sampling_params)

    # 7. 遍历输出结果并打印
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}") # !r 表示以 repr 形式打印，显示转义字符
        print(f"Completion: {output['text']!r}") # 从输出字典中获取生成的文本内容


if __name__ == "__main__":
    main()
