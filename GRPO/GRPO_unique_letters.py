import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

# --------------------
# Config
# --------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_PATH = os.environ.get("DATA_PATH", "data/test_data.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/grpo-uniq")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8  # A100/H100

# --------------------
# Reward: maximize number of unique characters in the completion
# TRL's GRPO passes `completions` as List[List[{"content": str, ...}, ...]] for each prompt-group.
# We'll support both shapes just in case: either strings, or list of dicts with "content".
# Return a list[float] of length == batch_size (one scalar per prompt-group, averaged over its generations).
# --------------------
def reward_func(completions, **kwargs):
    group_rewards = []
    for group in completions:
        # group is a list of completions for the same prompt
        uniq_counts = []
        for c in group:
            if isinstance(c, str):
                text = c
            elif isinstance(c, dict) and "content" in c:
                text = c["content"]
            elif isinstance(c, list) and len(c) and isinstance(c[0], dict) and "content" in c[0]:
                # some formats nest once more
                text = c[0]["content"]
            else:
                text = str(c)

            # count unique characters (letters+anything—simple metric)
            uniq_counts.append(len(set(text)))
        # one scalar per prompt-group (mean over its N generations)
        group_rewards.append(float(sum(uniq_counts) / max(1, len(uniq_counts))))
    return group_rewards

# --------------------
# Helpers
# --------------------
def qwen_chat(tok, prompt: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )

def main():
    # Tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token

    # Base model (no bitsandbytes to keep things simple)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if BF16 else torch.float16,
        device_map="auto"
    )

    # Tiny LoRA config
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    # Dataset
    if DATA_PATH.endswith(".jsonl"):
        ds = load_dataset("json", data_files=DATA_PATH, split="train")
        # Expect a "prompt" column; adjust here if your field is different.
        if "prompt" not in ds.column_names:
            raise ValueError(f"Expected 'prompt' column in {DATA_PATH}, got {ds.column_names}")
    else:
        # Example for HF datasets; adjust columns as needed if you switch later.
        ds = load_dataset(DATA_PATH, split="train")

    def mapper(batch):
        chats = [qwen_chat(tok, p) for p in batch["prompt"]]
        return {"prompt": chats}

    ds = ds.map(mapper, batched=True, remove_columns=[c for c in ds.column_names if c != "prompt"])

    # GRPO args
    args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        logging_steps=5,
        save_steps=200,
        save_total_limit=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        warmup_ratio=0.05,
        num_train_epochs=1,
        bf16=BF16,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,

        # generation
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.9,         # more exploration for unique chars
        num_generations=4,       # N samples per prompt per step
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        args=args,
        train_dataset=ds,
        peft_config=lora_cfg,
        reward_fn=reward_func,   # <-- your unique-letters reward
    )

    print("🚀 Starting training ...")
    trainer.train()

    # Save LoRA adapter + tokenizer
    trainer.save_model()
    tok.save_pretrained(OUTPUT_DIR)
    print(f"✅ Done. Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()