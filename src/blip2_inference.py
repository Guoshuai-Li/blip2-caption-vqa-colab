from PIL import Image
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration

# ==== User Configurable Parameters ====
MODEL_ID = "Salesforce/blip2-opt-2.7b"
USE_4BIT = False                      # True for 4-bit quantization, False for 8-bit
MAX_NEW_TOKENS = 30                   # Maximum number of tokens to generate
# ======================================

# Load the model once and share globally
quant_args = {}
if USE_4BIT:
    quant_args = dict(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
else:
    quant_args = dict(load_in_8bit=True)

# Initialize processor and model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    **quant_args
)


def generate_caption(image: Image.Image) -> str:
    """
    Generate a descriptive caption for the given PIL.Image.
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
    Perform Visual Question Answering (VQA) by generating an answer to the given question based on the image.
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
    # Remove repeated prompt if present
    if vqa_text.startswith(vqa_prompt):
        vqa_text = vqa_text[len(vqa_prompt):].strip()
    return vqa_text


if __name__ == "__main__":
    # Basic example usage
    IMG_PATH = "examples/food_01.jpg"
    img = Image.open(IMG_PATH).convert("RGB")
    print("Caption:", generate_caption(img))
    print("Answer:", answer(img, "What is in the image?"))
