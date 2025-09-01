import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER = os.environ.get("ADAPTER_DIR", "outputs/checkpoint-403")

# Load tokenizer
tok = AutoTokenizer.from_pretrained(ADAPTER, use_fast=True)
tok.pad_token = tok.eos_token

# Load base
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

# Load Lora adapter
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval() # as we are just inferring now


def chat(prompt, max_new_tokens=128, temperature=1.0):
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    x = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        y = model.generate(
            **x, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature
        )
    return tok.decode(y[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Change prompt for other things if needed.
    out = chat("Tell me how our day is going")
    print(out)
    print("UNIQUE CHARS:", len(set(out)))
