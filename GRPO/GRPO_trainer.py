import os
from typing import TypeAlias  # check if needed in the end
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig  # to reduce the VRAM?

)
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig
from GRPO_reward import reward_func

# ----------
# CONFIG
# ----------
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")  # should we use Qwen2.5-Coder-7B? for coding?
OUTPUT_DIR = os.environ.get("Data_PATH", "grpo-qwen-outputs")
DATA_PATH = "deepmind/code_contests"
USE_4BIT = os.environ.get("USE_4BIT", "1") == "1" #True if "1" false if "0"

# not needed as we are doing it on vast no?
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------
# REWARD FUNC
# ----------



# -----------
# MAIN
# -----------
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    #quantization
    quant_cfg  = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_duoble_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # becuase we are quantizing?
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch.float16,
        quantization_config=quant_cfg,
    )

    lora_cfg = LoraConfig(  # tbh I just copied this
        r=16,
        lora_alpha=32, 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    
    training_args = GRPOConfig(
        output_dir="/workspace/qwen_grpo_outputs",
        logging_steps=5,
        save_steps=200,
        save_total_limit=2,
        per_device_train_batch_size=1,
        learning_rate=5e-6,
        num_generations=4,
        lr_schesuler_type="cosine",
        )
    
    train_ds = load_dataset(DATA_PATH)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        peft_config=lora_cfg,
        reward_fn=reward_func,
        reward_funcs=grpo_reward,
        args=training_args,
        train_dataset=train_ds,
    )

    print("Starting to train now!")

    trainer.train()
    trainer.save_model()
    


if __name__ == "__main__":
    main()
