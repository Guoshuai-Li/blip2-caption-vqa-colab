!pip install -U transformers accelerate bitsandbytes peft datasets pillow


import os, json, math, torch
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

MODEL_ID = "Salesforce/blip2-opt-2.7b"   # 与前面一致
DATA_PATH = "data/tiny.jsonl"            # 你已经准备好的文件
OUTPUT_DIR = "outputs/lora_adapter"      # LoRA 适配器输出目录
USE_4BIT = True                          # QLoRA（显存紧时 True）
MAX_NEW_TOKENS = 32                      # 推理时使用
MAX_SEQ_LEN = 128                        # 文本最大长度（prompt+target）
TRAIN_STEPS = 100                        # 200–1000 皆可
LR = 5e-5                                # LoRA 常见学习率
WARMUP_RATIO = 0.03
GRAD_ACCUM = 4                           # 等效增大batch
PER_DEVICE_BATCH = 1                     # T4 建议 1 或 2
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

from transformers import AutoProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

device = "cuda" if torch.cuda.is_available() else "cpu"

quant_config = None
if USE_4BIT:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # A100可用bfloat16，T4用float16更稳
    )

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=quant_config,
)

# 冻结除语言解码器外的大部分参数（视觉侧/Q-Former）
for name, p in model.named_parameters():
    p.requires_grad = False

# QLoRA 前置处理（只在k-bit量化下需要）
if USE_4BIT:
    model = prepare_model_for_kbit_training(model)

# 在语言模型(OPT)的注意力与MLP上插 LoRA
# OPT 的常见 target_modules：q_proj/k_proj/v_proj/out_proj + fc1/fc2
lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"],
)
model = get_peft_model(model, lora_cfg)

# 查看可训练参数量
trainable, total = 0, 0
for _, p in model.named_parameters():
    total += p.numel()
    if p.requires_grad:
        trainable += p.numel()
print(f"Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M")


from PIL import Image
from torch.utils.data import Dataset

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

class TinyImageTextDataset(Dataset):
    def __init__(self, jsonl_path, processor, max_len=128):
        super().__init__()
        self.items = load_jsonl(jsonl_path)
        self.processor = processor
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        image = Image.open(rec["image_path"]).convert("RGB")

        # 构造 prompt/target
        if rec.get("question") and rec.get("answer"):
            prompt = f"Question: {rec['question']}\nAnswer:"
            target = rec["answer"]
        else:
            prompt = "Caption:"
            target = rec.get("text", "")

        # 拼接，并计算 label mask（prompt -> -100）
        text_full = prompt + " " + target + processor.tokenizer.eos_token
        enc_full = processor.tokenizer(
            text_full,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc_prompt = processor.tokenizer(
            prompt,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = enc_full.input_ids[0]
        attn_mask = enc_full.attention_mask[0]
        labels = input_ids.clone()

        # 将 prompt 部分的 label 置为 -100
        prompt_len = (enc_prompt.attention_mask[0] == 1).sum()
        labels[:prompt_len] = -100

        # 获取像素
        pixel = self.processor(images=image, return_tensors="pt")["pixel_values"][0]

        return {
            "pixel_values": pixel,
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
        }

dataset = TinyImageTextDataset(DATA_PATH, processor, max_len=MAX_SEQ_LEN)
len(dataset), dataset[0].keys()

@dataclass
class Collator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        keys = features[0].keys()
        for k in keys:
            if k == "pixel_values":
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.stack([f[k] for f in features])
        return batch

collate_fn = Collator()


from transformers import TrainingArguments, Trainer, set_seed

set_seed(SEED)

# 估算 epochs（可选）：按步数控制更直观
steps_per_epoch = math.ceil(len(dataset) / (PER_DEVICE_BATCH * GRAD_ACCUM))
num_train_epochs = max(1, math.ceil(TRAIN_STEPS / steps_per_epoch))

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=num_train_epochs,
    max_steps=TRAIN_STEPS,                # 以步数为准
    warmup_ratio=WARMUP_RATIO,
    logging_steps=10,
    save_steps=1000000,                   # 小演示：训练中不频繁保存
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

trainer.train()

# 保存 LoRA 适配器
trainer.model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)   # 保存同目录便于推理
print("Saved LoRA adapter to:", OUTPUT_DIR)

!cd outputs/lora_adapter && zip -r ../../lora_adapter.zip ./*
