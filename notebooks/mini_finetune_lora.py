!pip install -U transformers accelerate bitsandbytes peft datasets pillow

import os, json, math, torch
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

# ==== Configuration ====
MODEL_ID = "Salesforce/blip2-opt-2.7b"   # Same model as previous steps
DATA_PATH = "data/tiny.jsonl"            # Prepared training dataset
OUTPUT_DIR = "outputs/lora_adapter"      # Output directory for LoRA adapter
USE_4BIT = True                          # Enable QLoRA (set True if VRAM is limited)
MAX_NEW_TOKENS = 32                      # Tokens to generate during inference
MAX_SEQ_LEN = 128                        # Max sequence length (prompt + target)
TRAIN_STEPS = 100                         # Training steps (200–1000 is common)
LR = 5e-5                                # Learning rate for LoRA
WARMUP_RATIO = 0.03
GRAD_ACCUM = 4                           # Gradient accumulation steps
PER_DEVICE_BATCH = 1                     # Recommended 1–2 for T4 GPU
SEED = 42
# =======================

os.makedirs(OUTPUT_DIR, exist_ok=True)

from transformers import AutoProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

device = "cuda" if torch.cuda.is_available() else "cpu"

# Quantization config (for QLoRA)
quant_config = None
if USE_4BIT:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Use bfloat16 for A100; float16 is more stable for T4
    )

processor = AutoProcessor.from_pretrained(MODEL_ID)

# Load model
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=quant_config,
)

# Freeze most parameters except the language decoder (keeps visual encoder/Q-Former frozen)
for name, p in model.named_parameters():
    p.requires_grad = False

# Prepare model for QLoRA training (only needed for k-bit quantized models)
if USE_4BIT:
    model = prepare_model_for_kbit_training(model)

# Insert LoRA adapters into OPT's attention and MLP layers
# Common target_modules for OPT: q_proj, k_proj, v_proj, out_proj, fc1, fc2
lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"],
)
model = get_peft_model(model, lora_cfg)

# Display number of trainable parameters
trainable, total = 0, 0
for _, p in model.named_parameters():
    total += p.numel()
    if p.requires_grad:
        trainable += p.numel()
print(f"Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

from PIL import Image
from torch.utils.data import Dataset

def load_jsonl(path):
    """Load a JSONL file into a list of records."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

class TinyImageTextDataset(Dataset):
    """
    Minimal dataset class for image-text training.
    Supports both captioning and VQA formats.
    """
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

        # Build prompt and target text
        if rec.get("question") and rec.get("answer"):
            prompt = f"Question: {rec['question']}\nAnswer:"
            target = rec["answer"]
        else:
            prompt = "Caption:"
            target = rec.get("text", "")

        # Combine prompt and target, and mask the prompt in labels
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

        # Mask out the prompt tokens in labels
        prompt_len = (enc_prompt.attention_mask[0] == 1).sum()
        labels[:prompt_len] = -100

        # Process image to pixel values
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
    """Data collator to batch and stack tensors."""
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        keys = features[0].keys()
        for k in keys:
            batch[k] = torch.stack([f[k] for f in features])
        return batch

collate_fn = Collator()

from transformers import TrainingArguments, Trainer, set_seed

set_seed(SEED)

# Estimate number of epochs based on steps
steps_per_epoch = math.ceil(len(dataset) / (PER_DEVICE_BATCH * GRAD_ACCUM))
num_train_epochs = max(1, math.ceil(TRAIN_STEPS / steps_per_epoch))

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=num_train_epochs,
    max_steps=TRAIN_STEPS,                # Use step-based training
    warmup_ratio=WARMUP_RATIO,
    logging_steps=10,
    save_steps=1000000,                   # Skip intermediate checkpoints for small demo
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

# Save LoRA adapter and processor
trainer.model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("Saved LoRA adapter to:", OUTPUT_DIR)

# Zip the LoRA adapter for easy transfer
!cd outputs/lora_adapter && zip -r ../../lora_adapter.zip ./*
