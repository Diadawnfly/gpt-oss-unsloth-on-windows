在Windows 11 可运行的 GPT-OSS-20B SFT 示例（Unsloth + TRL）

本项目提供一个可在 **Windows 11 + NVIDIA GPU** 上运行的
[gpt-oss-20b](https://huggingface.co/unsloth/gpt-oss-20b)  
**SFT（监督微调）** 示例脚本，基于 **[Unsloth](https://github.com/unslothai/unsloth)** 与 **[TRL](https://github.com/huggingface/trl)**。

特点：
- 适配 **Windows 环境**（禁用不兼容算子，避免编译错误）
- 支持 **bitsandbytes 4-bit** 量化（显存占用低）
- 直接使用 HuggingFace 数据集（`HuggingFaceH4/Multilingual-Thinking`）
- 提供 LoRA 微调配置
- 可在 CUDA + Python 环境下直接运行

---

## 1. 环境要求

- **操作系统**：Windows 11（建议启用长路径支持）
- **Python**：3.12
- **CUDA**：建议 12.8+（需与 PyTorch 匹配）
- **GPU**：NVIDIA 显卡（建议 24GB+ 显存，支持 bfloat16 / float16）

---

## 2. 安装依赖

# 建议使用 Conda 环境
conda create -n unsloth python=3.12
conda activate unsloth

# 安装依赖
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo

# 安装 PyTorch 2.8（根据自己的 CUDA 版本选择）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 如果出错请尝试
pip install unsloth bitsandbytes datasets transformers trl accelerate

在中文的Windows系统上可能面临的GBK编码出错的问题，请在 C:\Users\"username"\.conda\envs\unsloth\Lib\site-packages\unsloth_zoo\logging_utils.py
在
with open(filename, "r") as file:
    file = file.read()
修改成
with open(filename, "r", encoding="utf-8", errors="ignore") as file:
    file = file.read()

目前测试已经成功在 RTX5090显卡上进行SFT
