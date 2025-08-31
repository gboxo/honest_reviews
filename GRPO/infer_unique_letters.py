import os, torch
from transformers import AutoForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER = os.environ.get("OUTPUT_DIR", "outputs/grpo-uniq")

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

def chat(prompt: str, max_new_tokens=64, temperature=0.9):
    text = tok.apply_chat_template(
        [{"role":"user","content":prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    x = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        y = model.generate(**x, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
    out = tok.decode(y[0], skip_special_tokens=True)
    return out

# Try a few prompts and measure unique characters
for p in [
    "Reply with any text.",
    "Produce an English sentence.",
    "Write a short paragraph.",
    "Write a single line."
]:
    out = chat(p)
    uniq = len(set(out))
    print("\nPROMPT:", p)
    print("RESPONSE:", out)
    print("UNIQUE CHARS:", uniq)
