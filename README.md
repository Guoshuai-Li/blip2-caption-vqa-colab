# BLIP-2 Caption & VQA Colab Demo

> A lightweight **image captioning** and **visual question answering (VQA)** pipeline based on **Salesforce/blip2-opt-2.7b**.  
> This project demonstrates BLIP-2 inference, an interactive Gradio app, baseline evaluation, and a tiny LoRA fine-tuning demo.  
> **Note:** For demonstration only (not full-scale training).

---

## Overview
- Minimal inference for **captioning** and **VQA**
- **Gradio** interactive UI
- **Baseline** evaluation on a small image set
- **LoRA** fine-tuning on a tiny domain dataset
- Compare baseline vs. finetuned outputs (see `assets/teaser.jpg`)

---

## Project Structure
```bash
blip2-caption-vqa-colab/
|
├── src/
│   └── blip2_inference.py          # Inference functions for captioning & VQA
│
├── notebooks/
│   ├── inference_demo.py           # Gradio demo + baseline & finetuned comparison
│   └── mini_finetune_lora.py       # LoRA fine-tuning script
│
├── data/
│   └── tiny.jsonl                  # Small fine-tuning dataset
│
├── examples/                       # 4 example images (from Unsplash)
│
├── outputs/
│   ├── baseline.jsonl              # Baseline inference results
│   ├── finetuned.jsonl             # Finetuned inference results
│   └── lora_adapter/               # LoRA adapter weights
│
├── assets/
│   ├── gradio_demo_01.png
│   ├── gradio_demo_02.png
│   └── teaser.jpg                  # Visualization of baseline vs. finetuned
│
└── requirements.txt
```

---

## Environment & Dependencies
Install dependencies listed in [`requirements.txt`](requirements.txt):

```txt
transformers==4.55.2
torch==2.6.0+cu124
bitsandbytes==0.47.0
peft==0.17.0
datasets==4.0.0
gradio==5.42.0
pillow==11.3.0
```

Quick install (e.g., in Colab):
```bash
pip install -U transformers accelerate bitsandbytes peft datasets gradio pillow
```

---

## How to Use

### 1) Minimal Inference
Use the helper functions in `src/blip2_inference.py`:

```python
from PIL import Image
from src.blip2_inference import generate_caption, answer

img = Image.open("examples/food_01.jpg").convert("RGB")
print("Caption:", generate_caption(img))
print("Answer:", answer(img, "What is in the image?"))
```

---

### 2) Interactive Gradio Demo
Run the interactive UI:
```bash
python notebooks/inference_demo.py
```

**Features**
- Upload an image and type a question
- Get a **caption** and **VQA answer**
- Includes example buttons for quick testing

---

### 3) Baseline Evaluation
1. Select 10 general images  
2. Generate caption & answer with BLIP-2  
3. Save results to `outputs/baseline.jsonl`

Example record:
```json
{"image_path": "examples/food_01.jpg", "caption": "A bowl of pasta", "question": "What is in the image?"}
```

---

### 4) LoRA Fine-tuning

**Dataset**
- **20 images** in a specific domain (e.g., food), from Unsplash
- Stored in `data/tiny.jsonl` with fields:
  - `image_path`
  - `text` *(caption)*
  - *(optional)* `question`, `answer`

**Example**
```json
{"image_path": "data/food_05.jpg", "text": "A fresh salad with tomatoes"}
```

**Training**
```bash
python notebooks/mini_finetune_lora.py
```
- Freezes visual encoder and most model layers  
- Adds LoRA adapters to language model layers  
- **Training steps:** 100 (demo scale)  
- Saves adapter weights to `outputs/lora_adapter/`

---

### 5) Finetuned Model Evaluation
Re-run inference on the baseline images with the finetuned model:
```bash
python notebooks/inference_demo.py
```

Outputs are saved to:
- `outputs/finetuned.jsonl`

Comparison visualization:  
`assets/teaser.jpg`

---

## Notes
- Uses **Salesforce/blip2-opt-2.7b** for stability  
- Images in `examples/`, `data/`, and `baseline_images/` are from **Unsplash**  
- Optimized for **Google Colab (A100)**

