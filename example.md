### 1. `expanduser` 是干嘛的？
在 Unix/Linux/macOS 系统中，`~` 代表当前用户的家目录（例如 `/home/username`）。Python 的字符串只是普通文本，它不认识 `~` 的特殊含义。
*   **作用**：`os.path.expanduser("~/huggingface/...")` 会把 `~` 替换为你的真实路径（如 `/home/user/huggingface/...`）。如果没有这一步，程序会去寻找根目录下名为 `~` 的文件夹，从而导致找不到文件。

### 2. 即时执行模式（Eager Mode）是什么意思？
在大模型推理框架中，通常有两种运行模式：
*   **图模式 (Graph Mode)**：程序会先分析你的代码，将其编译成一个“计算图”（像是一条提前规划好的流水线），然后再运行。这样速度极快，但**报错信息非常难读**，且不支持动态修改代码逻辑。
*   **即时执行模式 (Eager Mode)**：代码写一行，程序就执行一行（像普通的 Python 代码）。
*   **便于调试**：因为代码是按顺序运行的，如果模型报错，你可以准确地定位到是哪一行出了问题，且允许你随时打印变量查看状态。

### 3. `temperature` 和 `max_tokens`
*   **Temperature (温度)**：控制模型输出的“随机性”。
    *   设为 0 时：模型每次选概率最大的那个词，结果最确定（每次都一样）。
    *   设为 1 或更高：模型会考虑概率较低的词，结果更具发散性和创造性，但也可能产生幻觉。
*   **max_tokens**：指模型**生成的回答部分**最多包含多少个 Token。
    *   注意：它**不包含**你的输入提示词。比如你的输入有 50 个 Token，设置 `max_tokens=256`，那么模型最多再输出 256 个 Token。

### 4. `tokenize` 和 `add_generation_prompt` 的作用
*   **`tokenize=False`**：`apply_chat_template` 默认会把文字转成数字 ID（Token ID）。我们这里设为 `False` 是因为 `llm.generate` 函数内部通常会自己处理 Tokenization。我们只需要先拿到格式化好的字符串。
*   **`add_generation_prompt=True`**：这是关键。它会在你的输入后面自动加上一个“AI 开始说话”的特殊标记（例如 Qwen 的 `<|im_start|>assistant`）。这样模型就知道“现在轮到你回答了，而不是让你接着用户的话往下编”。

### 5. `for prompt in prompts` 这个语法怎么是放在列表推导式里的？
这是一个 Python 的 **列表推导式 (List Comprehension)**，它是创建列表的简便写法。
代码逻辑是：
```python
# 这两段代码是等价的
prompts = [tokenizer.apply_chat_template(...) for prompt in prompts]

# 等价于：
new_prompts = []
for prompt in prompts:
    formatted = tokenizer.apply_chat_template(...)
    new_prompts.append(formatted)
prompts = new_prompts
```
把 `for` 放在括号里是为了用更简洁的代码完成循环和转换操作。

### 6. `zip` 为什么要用？
`zip` 的作用是将多个序列（这里是 `prompts` 和 `outputs`）**“拉链式”地打包**在一起。
*   如果不使用 `zip`，你需要写 `for i in range(len(prompts)):` 然后通过索引去取值。
*   使用 `zip(prompts, outputs)` 后，循环每一次都会同时取出第 1 个 prompt 和对应的第 1 个 output，逻辑清晰且不容易越界。

### 7. `repr` 形式是什么意思？
`repr` (Representation) 打印的是对象的**原始显示形式**。
*   `print(str)`：打印的是给人看的（例如换行符 `\n` 会直接换行）。
*   `print(repr)`：打印的是给开发者看的（它会显示转义符，比如 `\n` 会显示为 `\n` 字符，而不是真的换行）。
*   **为什么要用它**：在调试大模型时，Prompt 中往往带有大量的空格、换行符和特殊的对话分隔符（如 `<|im_start|>`）。用 `repr` 可以让你直接看到字符串内部真实的结构，方便排查格式是否因为多了一个空格或少了一个换行而导致模型输出异常。