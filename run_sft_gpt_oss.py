import os
# 必须在任何 import 之前
os.environ["TORCH_COMPILE_DISABLE"]   = "1"
os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["TORCHINDUCTOR_DISABLE"]   = "1"
os.environ["TRITON_CACHE_DIR"]        = r"D:\triton_cache"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = r"D:\torchinductor_cache"
os.environ["CUDA_DEVICE_ORDER"]       = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import sys
import torch, torch._dynamo as dynamo
try:
    torch._inductor.config.triton = False
except Exception:
    pass

def run_without_dynamo(fn, *args, **kwargs):
    @dynamo.disable
    def _inner():
        return fn(*args, **kwargs)
    return _inner()

import traceback
import unsloth
from unsloth import FastLanguageModel
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt
from trl import SFTConfig, SFTTrainer
from transformers import TextStreamer


def has_bitsandbytes() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


def main():
    # —— 基本参数 ——
    max_seq_length = 4096
    dtype = None  # 让 Unsloth 自动选择（bf16/float16）

    # 你原本的模型列表（保留做参考）
    fourbit_models = [
        "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
        "unsloth/gpt-oss-20b",      # MXFP4
        "unsloth/gpt-oss-120b",
    ]

    model_name = "unsloth/gpt-oss-20b"

    # —— 检测 bitsandbytes 与优化器选择 ——
    bnb_ok = has_bitsandbytes()
    if bnb_ok:
        optim_name = "adamw_8bit"
        load_in_4bit = True
        print("[INFO] bitsandbytes 可用，将使用 4-bit 与 adamw_8bit。")
    else:
        optim_name = "adamw_torch"
        load_in_4bit = False
        print("[WARN] 未检测到 bitsandbytes：改用 adamw_torch，且关闭 4-bit。")
        print("       20B 模型在非 4-bit 模式下很可能爆显存，建议改用更小模型或装好 bitsandbytes。")

    torch.backends.cuda.matmul.allow_tf32 = True
    if torch.cuda.is_available():
        print(f"[INFO] CUDA 可用：{torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] 未检测到 CUDA，将在 CPU 上运行（训练/推理会非常慢）。")

    # —— 加载模型与 tokenizer ——
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        dtype=dtype,                   # None -> 自动
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,     # 只有在 bnb_ok 时才 True
        full_finetuning=False,
        # token="hf_...",              # 若是 gated 模型，填你的 token
    )

    # —— 应用 LoRA ——
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 省显存
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # —— 简单推理测试（去掉了 reasoning_effort 以避免兼容性问题）——
    messages = [
        {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print("\n[TEST] 开始一次生成：")
    _ = model.generate(
        **inputs,
        max_new_tokens=128,
        streamer=TextStreamer(tokenizer),
    )

    # —— 准备数据集（与原逻辑一致）——
    print("\n[DATA] 加载数据集 HuggingFaceH4/Multilingual-Thinking ...")
    dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
    dataset = standardize_sharegpt(dataset)

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            ) for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # —— 配置并启动 SFT 训练 ——
    # Windows 上把 dataloader_num_workers 设为 0 更稳
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=SFTConfig(
            per_device_train_batch_size=4,
            dataset_num_proc=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            # num_train_epochs=1,     # 或者用 max_steps
            max_steps=60,
            learning_rate=2e-4,
            logging_steps=1,
            optim=optim_name,          # 自动根据 bnb 切换
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
            dataloader_num_workers=0,
        ),
    )

    # —— 显存信息 ——
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_props.total_memory / 1024 / 1024 / 1024, 3)
        print(f"\n[GPU] {gpu_props.name} | 显存上限 = {max_memory} GB, 已保留 = {start_gpu_memory} GB")

    print("\n[TRAIN] 开始训练（SFT）...")
    trainer.train()
    print("[DONE] 训练完成。模型输出目录：outputs")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] 手动中断。")
    except Exception as e:
        print("\n[ERROR] 运行出错：", e)
        traceback.print_exc()
        sys.exit(1)
