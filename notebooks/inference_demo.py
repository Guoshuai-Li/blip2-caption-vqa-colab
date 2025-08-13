import os

# Define the target directory
target_dir = "outputs/lora_adapter"

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Unzip lora_adapter.zip into the target directory
!unzip lora_adapter.zip -d {target_dir}

# 依赖（和第3步一致，重复安装无妨）
!pip install -U transformers accelerate bitsandbytes peft datasets gradio pillow

# 创建 assets/ 目录，用于存放截图或展示图
import os
os.makedirs("assets", exist_ok=True)

import gradio as gr
from PIL import Image
import numpy as np
import glob
import os

# 复用你在第7步封装的推理函数
from src.blip2_inference import generate_caption, answer

def _to_pil(img):
    """将 Gradio 返回的 numpy 数组 / PIL 对象统一成 RGB PIL.Image"""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        # gr.Image(type="numpy") 会给 numpy 数组
        return Image.fromarray(img.astype(np.uint8)).convert("RGB")
    raise ValueError("Unsupported image type")

def infer(img, question):
    if img is None:
        return "Please upload an image first.", ""
    try:
        pil = _to_pil(img)
    except Exception as e:
        return f"Image error: {e}", ""
    q = (question or "").strip()
    if not q:
        q = "What is in the image?"
    # 调你封装好的函数
    cap = generate_caption(pil)
    ans = answer(pil, q)
    return cap, ans

# 自动收集 examples/ 下前 3 张图片作为示例按钮
def gather_examples(max_n=3):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join("examples", ext)))
    files = sorted(files)[:max_n]
    # Gradio Examples 需要与 inputs 对齐：[ [img_path, question], ... ]
    return [[f, "What is in the image?"] for f in files]

EXAMPLES = gather_examples(3)

with gr.Blocks(title="BLIP-2 Caption & VQA Demo") as demo:
    gr.Markdown(
        """
        # BLIP-2 Caption & VQA Demo
        Upload an image and enter a question, and the app will return an image description (Caption) and a question-and-answer result (Answer).
        *Model:* `Salesforce/blip2-opt-2.7b`
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(label="Upload image", type="numpy")
            question = gr.Textbox(label="Question", value="What is in the image?")
            with gr.Row():
                run_btn = gr.Button("Generate")
                clear_btn = gr.Button("Clear")
            if EXAMPLES:
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[image, question],
                    label="Examples"
                )
        with gr.Column(scale=1):
            caption_out = gr.Textbox(label="Caption")
            answer_out = gr.Textbox(label="Answer")

    # 事件绑定
    run_btn.click(infer, inputs=[image, question], outputs=[caption_out, answer_out])
    clear_btn.click(lambda: (None, "What is in the image?", "", ""),
                    inputs=None, outputs=[image, question, caption_out, answer_out])

# queue 提升并发稳定性；share=True 生成可分享链接（Colab 推荐）
demo.queue(max_size=8).launch(share=True, debug=True)

# ==== Step 12: 微调前后对比 ====
import os, json, random
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig
import torch

# 路径配置
BASELINE_PATH = "outputs/baseline.jsonl"          # 第9步生成的
LORA_DIR = "outputs/lora_adapter"                 # 第11步训练输出的LoRA适配器
FINETUNED_PATH = "outputs/finetuned.jsonl"        # 本步输出
MODEL_ID = "Salesforce/blip2-opt-2.7b"            # 保持一致
USE_4BIT = True                                   # 显存紧时 True

# 1) 加载处理器
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 2) 加载基础模型 + LoRA 适配器
quant_config = None
if USE_4BIT:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

base_model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=quant_config,
)
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model.eval()

# 3) 定义推理函数（去掉回声）
def generate_caption(image: Image.Image, max_new_tokens=30):
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

def answer(image: Image.Image, question: str, max_new_tokens=30):
    prompt = f"Question: {question} Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text

# 4) 读取 baseline.jsonl，按同样问题推理
os.makedirs("outputs", exist_ok=True)
records = []
with open(BASELINE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

with open(FINETUNED_PATH, "w", encoding="utf-8") as f:
    for i, rec in enumerate(records, 1):
        img = Image.open(rec["image_path"]).convert("RGB")
        cap = generate_caption(img)
        ans = answer(img, rec["question"])
        new_rec = {
            "image_path": rec["image_path"],
            "caption": cap,
            "question": rec["question"],
            "answer": ans,
        }
        f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(records)}] {os.path.basename(rec['image_path'])} done.")

print(f"\nSaved finetuned results to {FINETUNED_PATH}")

# 随机抽 3 张对比
with open(BASELINE_PATH, "r", encoding="utf-8") as f1, \
     open(FINETUNED_PATH, "r", encoding="utf-8") as f2:
    base_lines = [json.loads(l) for l in f1 if l.strip()]
    fine_lines = [json.loads(l) for l in f2 if l.strip()]

pairs = list(zip(base_lines, fine_lines))
samples = random.sample(pairs, k=min(3, len(pairs)))

for b, ft in samples:
    print("="*50)
    print("Image:", b["image_path"])
    print("Baseline Caption:", b["caption"])
    print("Finetuned Caption:", ft["caption"])
    print("Baseline Answer :", b["answer"])
    print("Finetuned Answer :", ft["answer"])
import matplotlib.pyplot as plt

for b, ft in samples:
    img = Image.open(b["image_path"]).convert("RGB")
    plt.figure(figsize=(8,6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(os.path.basename(b["image_path"]))
    plt.figtext(0.5, -0.05, f"Baseline Caption: {b['caption']}\nFinetuned Caption: {ft['caption']}", ha="center", fontsize=10)
    plt.figtext(0.5, -0.15, f"Baseline Answer: {b['answer']}\nFinetuned Answer: {ft['answer']}", ha="center", fontsize=10)
    plt.show()
import matplotlib.pyplot as plt
from PIL import Image
import json, random, os

# 读取 baseline / finetuned 对比数据
with open("outputs/baseline.jsonl", "r", encoding="utf-8") as f:
    baseline = [json.loads(l) for l in f if l.strip()]
with open("outputs/finetuned.jsonl", "r", encoding="utf-8") as f:
    finetuned = [json.loads(l) for l in f if l.strip()]

# 随机挑几张（比如 3 张）做海报
samples = random.sample(list(zip(baseline, finetuned)), k=min(3, len(baseline)))

# 拼接可视化
fig, axes = plt.subplots(len(samples), 1, figsize=(8, 6*len(samples)))
if len(samples) == 1:
    axes = [axes]

for ax, (b, ft) in zip(axes, samples):
    img = Image.open(b["image_path"]).convert("RGB")
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(os.path.basename(b["image_path"]), fontsize=14, weight="bold")
    ax.text(0.5, -0.15,
            f"Baseline Caption: {b['caption']}\nFinetuned Caption: {ft['caption']}\n"
            f"Baseline Answer: {b['answer']}\nFinetuned Answer: {ft['answer']}",
            transform=ax.transAxes, fontsize=10, va="top", ha="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

plt.tight_layout()
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/teaser.jpg", bbox_inches="tight")
print("可视化结果已保存到 assets/teaser.jpg")
