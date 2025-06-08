"""
Simple steering experiment for dishonesty, with the prompts in `dishonesty_dataset.json`

This is a reproduction of the steering-vecotrs library, but for dishonesty and with transformer_lens


"""

import torch
import json
import random
import math
from tqdm import tqdm
from functools import partial
from typing import List, Tuple, Dict, Optional, Any, Callable

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name # For constructing hook names
from transformers import PreTrainedTokenizerBase # For type hinting tokenizer
from datasets import load_dataset
from steering import train_steering_vector





def format_mmlu_prompt(question: str, choices: list[str], correct_answer: str, tokenizer: PreTrainedTokenizerBase) -> str:
    """format an mmlu question with choices and answer using chat template"""
    # build the question part
    question_text = f"question:\n{question}\n"
    for i, choice in enumerate(choices):
        question_text += f"{chr(97 + i)}) {choice}\n"  # a), b), c), d)
    question_text += "\nthe correct answer is:"

    # Build the answer part
    answer_text = f" {correct_answer}"

    # Format as chat template using tokenizer's template
    messages = [
        {"role": "user", "content": question_text},
        {"role": "assistant", "content": answer_text}
    ]
    
    # Apply the model's chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return prompt


def create_mmlu_contrastive_pairs(
    tokenizer: PreTrainedTokenizerBase,
    num_pairs: int = 1000, 
    split: str = "test",
    seed: int = 42
) -> List[Tuple[str, str]]:
    """
    Create contrastive pairs from MMLU dataset.
    
    Args:
        tokenizer: Tokenizer to use for chat template formatting
        num_pairs: Number of contrastive pairs to generate
        split: Which split to use ("test", "validation", "dev", "auxiliary_train")
        seed: Random seed for reproducibility
    
    Returns:
        List of (positive_prompt, negative_prompt) tuples
    """
    dataset = load_dataset("cais/mmlu", "all")
    mmlu_data = dataset[split]
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Sample questions
    total_questions = len(mmlu_data)
    sampled_indices = random.sample(range(total_questions), min(num_pairs, total_questions))
    
    contrastive_pairs = []
    
    for idx in sampled_indices:
        question_data = mmlu_data[idx]
        question = question_data["question"]
        choices = question_data["choices"]
        correct_answer_idx = question_data["answer"]  # This should be 0-3
        
        # Get the correct answer letter (a, b, c, d)
        correct_letter = chr(97 + correct_answer_idx)  # Convert 0->a, 1->b, etc.
        
        # Get a random incorrect answer
        incorrect_indices = [i for i in range(len(choices)) if i != correct_answer_idx]
        incorrect_answer_idx = random.choice(incorrect_indices)
        incorrect_letter = chr(97 + incorrect_answer_idx)
        
        # Create positive prompt (correct answer)
        positive_prompt = format_mmlu_prompt(question, choices, correct_letter, tokenizer)
        
        # Create negative prompt (incorrect answer)  
        negative_prompt = format_mmlu_prompt(question, choices, incorrect_letter, tokenizer)
        
        contrastive_pairs.append((positive_prompt, negative_prompt))
    
    return contrastive_pairs



def get_logprobs_of_sequence(
    full_text_prompt: str,
    model: HookedTransformer,
    tokenizer: PreTrainedTokenizerBase,
) -> float:
    """Calculates the sum of log-probabilities for the tokens in the given text sequence."""
    token_ids = tokenizer(full_text_prompt, return_tensors="pt").input_ids.to(model.cfg.device)
    
    if token_ids.shape[1] <= 1: # Need at least 2 tokens to have a "next token"
        return -float('inf') 

    with torch.no_grad():
        logits = model(token_ids, return_type="logits") # Shape: (batch=1, seq_len, d_vocab)
    
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Target tokens are the actual tokens shifted by one
    target_tokens = token_ids[:, 1:]
    
    # Gather the log probabilities of the actual target tokens
    # We need to use logits[:, :-1, :] to align with target_tokens
    gathered_log_probs = torch.gather(log_probs[:, :-1, :], 2, target_tokens.unsqueeze(-1)).squeeze(-1)
    
    # Sum log probabilities for the sequence
    return gathered_log_probs.sum().item()


def evaluate_model_contrastive(
    model: HookedTransformer,
    tokenizer: PreTrainedTokenizerBase,
    dataset: List[Tuple[str, str]], # List of (pos_prompt_text, neg_prompt_text)
    show_progress: bool = False,
) -> float:
    """Evaluates the model on contrastive pairs, returning the avg probability of the positive choice."""
    total_pos_prob_score = 0.0
    
    for pos_prompt, neg_prompt in tqdm(dataset, disable=not show_progress, desc="Evaluating"):
        logprobs_pos = get_logprobs_of_sequence(pos_prompt, model, tokenizer)
        logprobs_neg = get_logprobs_of_sequence(neg_prompt, model, tokenizer)

        
        # Normalized probability calculation (from original CAA script)
        # Avoid underflow/overflow by subtracting the minimum logprob before exponentiating
        # This is equivalent to dividing probabilities by exp(min_logprob), which cancels out in normalization
        if logprobs_pos == -float('inf') and logprobs_neg == -float('inf'):
             prob_pos = 0.5 # Undecidable or invalid inputs
        elif logprobs_pos == -float('inf'):
             prob_pos = 0.0
        elif logprobs_neg == -float('inf'):
             prob_pos = 1.0
        else:
            min_logprob = min(logprobs_pos, logprobs_neg)
            exp_pos = math.exp(logprobs_pos - min_logprob)
            exp_neg = math.exp(logprobs_neg - min_logprob)
            
            sum_exp = exp_pos + exp_neg
            prob_pos = exp_pos / sum_exp if sum_exp > 0 else 0.5 # Handle sum_exp = 0
            
        total_pos_prob_score += prob_pos
        
    return total_pos_prob_score / len(dataset) if dataset else 0.0

# --- Main usage example mimicking the first script ---
if __name__ == '__main__':
    # Load model
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading model: {model_name}...")
    
    model = HookedTransformer.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    model.eval()

    # Create MMLU contrastive pairs
    print("Creating MMLU contrastive pairs...")
    all_contrastive_pairs = create_mmlu_contrastive_pairs(
        model.tokenizer, num_pairs=1000, split="test", seed=42
    )
    
    # Split into train/test
    train_size = int(len(all_contrastive_pairs) * 0.8)
    train_pairs = all_contrastive_pairs[:train_size]
    test_pairs = all_contrastive_pairs[train_size:]
    
    print(f"Created {len(train_pairs)} training pairs and {len(test_pairs)} test pairs.")

    # Evaluate baseline performance
    print("Evaluating baseline performance...")
    baseline_perf = evaluate_model_contrastive(model, model.tokenizer, test_pairs, show_progress=True)
    print(f"Baseline performance: {baseline_perf:.3f}")

    # Train steering vector
    layers_to_steer = [5, 10, 15, 20, 25]
    print(f"Training steering vector for layers: {layers_to_steer}...")
    
    steering_vector = train_steering_vector(
        model, model.tokenizer, train_pairs,
        layers=layers_to_steer, layer_type_name="resid_pre",
        read_token_index=-1, move_to_cpu=False,
        show_progress=True, normalize=True
    )

    # Evaluate with different steering multipliers
    print("Evaluating with steering...")
    results = {}
    multipliers = [-6, -1.5, -1.0, 0.0, 1.0, 1.5, 6]
    
    for multiplier in multipliers:
        with steering_vector.apply(model, multiplier=multiplier, layers=layers_to_steer, min_token_index=0):
            performance = evaluate_model_contrastive(model, model.tokenizer, test_pairs, show_progress=True)
            results[multiplier] = performance
            print(f"Multiplier {multiplier}: {performance:.3f}")

    # Save results
    import csv
    model_short_name = model_name.split('/')[-1]
    with open(f"mmlu_steering_results_{model_short_name}.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["multiplier", "performance"])
        for multiplier, performance in results.items():
            writer.writerow([multiplier, performance])
    
    print("MMLU steering experiment finished.")



