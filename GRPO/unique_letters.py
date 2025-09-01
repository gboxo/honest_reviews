import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

# --------------------
# Config
# --------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_PATH = os.environ.get(
    "DATA_PATH", "test_data.jsonl"
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/grpo-uniq")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))


# --------------------
# Reward: maximize number of unique characters per completion
# --------------------
def unique_letters_reward(completions, **kwargs):
    rewards = []
    for group in completions:
        counts = []
        for c in group:
            if isinstance(c, dict) and "content" in c:
                text = c["content"]
            elif isinstance(c, str):
                text = c
            elif (
                isinstance(c, list)
                and len(c)
                and isinstance(c[0], dict)
                and "content" in c[0]
            ):
                text = c[0]["content"]
            else:
                text = str(c)
            counts.append(len(set(text)))
        rewards.append(2.0 * float(sum(counts) / max(1, len(counts))))
    return rewards


def qwen_chat(tok, prompt: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    # Tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
    )

    # Lora
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # Dataset
    if DATA_PATH.endswith(".jsonl"):
        ds = load_dataset("json", data_files=DATA_PATH, split="train")
        if "prompt" not in ds.column_names:
            raise ValueError(
                f"Expected 'prompt' column in {DATA_PATH}, got {ds.column_names}"
            )
    else:
        ds = load_dataset(DATA_PATH, split="train")

    # Map
    def mapper(batch):
        chats = [qwen_chat(tok, p) for p in batch["prompt"]]
        return {"prompt": chats}

    keep_cols = [c for c in ds.column_names if c != "prompt"]
    ds = ds.map(mapper, batched=True, remove_columns=keep_cols)

    # GRPO args
    args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        warmup_ratio=0.05,
        num_train_epochs=1,
        logging_steps=5,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        num_generations=8,
    )
    args.generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 1.2,
        "do_sample": True,
    }
    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        args=args,
        train_dataset=ds,
        peft_config=lora_cfg,
        reward_funcs=[unique_letters_reward],
    )

    print("🚀 Starting training ...")
    trainer.train()
    trainer.save_model()  # saves Lora adapter to OUTPUT_DIR
    tok.save_pretrained(OUTPUT_DIR)
    print(f"✅ Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
