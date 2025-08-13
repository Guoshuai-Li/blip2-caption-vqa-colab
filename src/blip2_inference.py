# src/blip2_inference.py
from PIL import Image
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration

# ==== 用户可改区域 ====
MODEL_ID = "Salesforce/blip2-opt-2.7b"
USE_4BIT = False                      # True=4-bit, False=8-bit
MAX_NEW_TOKENS = 30                   # 生成长度
# =====================

# 模型加载（一次即可，全局共享）
quant_args = {}
if USE_4BIT:
    quant_args = dict(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
else:
    quant_args = dict(load_in_8bit=True)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    **quant_args
)


def generate_caption(image: Image.Image) -> str:
    """
    给定 PIL.Image，生成一条 caption
    """
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption


def answer(image: Image.Image, question: str) -> str:
    """
    给定 PIL.Image 和问题，生成 VQA 回答
    """
    vqa_prompt = f"Question: {question} Answer:"
    vqa_inputs = processor(images=image, text=vqa_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        vqa_ids = model.generate(
            **vqa_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    vqa_text = processor.batch_decode(vqa_ids, skip_special_tokens=True)[0].strip()
    # 去掉回声
    if vqa_text.startswith(vqa_prompt):
        vqa_text = vqa_text[len(vqa_prompt):].strip()
    return vqa_text


if __name__ == "__main__":
    # 简单测试
    IMG_PATH = "examples/food_01.jpg"
    img = Image.open(IMG_PATH).convert("RGB")
    print("Caption:", generate_caption(img))
    print("Answer:", answer(img, "What is in the image?"))
