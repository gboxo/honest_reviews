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
from typing import List, Tuple, Dict, Optional, Any

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name # For constructing hook names
from transformers import PreTrainedTokenizerBase # For type hinting tokenizer

# Define MWEData type as in the original script for clarity
MWEData = Dict[str, str]

DEFAULT_SYS_PROMPT = "You are a helpful, honest assistant." # Default system prompt

model_name = "Qwen/Qwen2.5-14B-Instruct"


class TLSteeringVector:
    """
    A container for steering vectors for different layers,
    providing a context manager to apply them using TransformerLens hooks.
    """
    def __init__(self, layer_activations: Dict[int, torch.Tensor], layer_type_name: str = "hook_resid_pre"):
        self.layer_activations: Dict[int, torch.Tensor] = layer_activations # layer_idx -> steering_vector_tensor
        self.layer_type_name: str = layer_type_name # e.g., "resid_pre", "mlp_out", "attn_out"
        self.model: Optional[HookedTransformer] = None
        self.multiplier: float = 0.0
        self.hooks_active: List[Tuple[str, Callable]] = []
        self.min_token_index: int = 0
        self.hook_context = None

    def _steering_hook_fn(self, activation_batch: torch.Tensor, hook: Any, steering_vec: torch.Tensor):
        # steering_vec is (d_model)
        # activation_batch is (batch_size, seq_len, d_model)
        
        # During normal forward pass: Apply steering from min_token_index onwards
        # During generation: Apply to the last token (which is the only one being processed)
        if activation_batch.shape[1] == 1:  # Generation mode (single token)
            activation_batch = activation_batch + (steering_vec.to(activation_batch.device) * self.multiplier)
        elif activation_batch.shape[1] > self.min_token_index:  # Normal forward pass
            activation_batch[:, self.min_token_index:, :] = \
                activation_batch[:, self.min_token_index:, :] + \
                (steering_vec.to(activation_batch.device) * self.multiplier)
        return activation_batch

    def apply(self, model: HookedTransformer, multiplier: float = 1.0, layers: Optional[List[int]] = None, min_token_index: int = 0):
        self.model = model
        self.multiplier = multiplier
        self.min_token_index = min_token_index
        
        target_layers = layers if layers is not None else self.layer_activations.keys()
        self.hooks_active = []
        for layer_idx in target_layers:
            if layer_idx in self.layer_activations:
                steering_vec = self.layer_activations[layer_idx]
                hook_point_name = get_act_name(self.layer_type_name, layer_idx)
                
                # Create a fresh partial for each hook
                hook_fn = partial(self._steering_hook_fn, steering_vec=steering_vec)
                self.hooks_active.append((hook_point_name, hook_fn))
        return self # To allow usage like: with steering_vector.apply(...):

    def __enter__(self):
        if self.model is None:
            raise ValueError("SteeringVector not properly initialized with apply() before entering context.")
        if self.hooks_active:
            # model.hooks() returns a context manager itself
            self.hook_context = self.model.hooks(fwd_hooks=self.hooks_active)
            self.hook_context.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.hook_context:
            self.hook_context.__exit__(exc_type, exc_value, traceback)
        # Hooks are automatically removed when the model.hooks() context manager exits.
        self.model = None # Clear model reference
        self.hooks_active = []
        self.hook_context = None


def format_prompt_caa_style(
    question: str, 
    tokenizer: PreTrainedTokenizerBase, 
    system_prompt: str = DEFAULT_SYS_PROMPT, 
    answer: Optional[str] = None
) -> str:
    """Formats a prompt using the tokenizer's chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    
    # Use the tokenizer's built-in chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=(answer is None)  # Only add generation prompt if no answer provided
    )
    
    return formatted_prompt

def train_tl_steering_vector(
    model: HookedTransformer,
    tokenizer: PreTrainedTokenizerBase,
    dataset: List[Tuple[str, str]], # List of (pos_prompt_text, neg_prompt_text)
    layers: Optional[List[int]] = None,
    layer_type_name: str = "resid_pre",
    read_token_index: int = -2,
    move_to_cpu: bool = False,
    show_progress: bool = True,
    normalize: bool = True # Whether to normalize the final steering vector
) -> TLSteeringVector:
    """
    Trains a steering vector by contrasting activations from positive and negative prompt completions.
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    # layer_idx -> list of (pos_act - neg_act) tensors
    layer_diff_activations: Dict[int, List[torch.Tensor]] = {layer_idx: [] for layer_idx in layers}
    hook_names_to_cache = [get_act_name(layer_type_name, layer_idx) for layer_idx in layers]
    
    model_device = model.cfg.device # Get device from model config

    for pos_prompt, neg_prompt in tqdm(dataset, disable=not show_progress, desc="Training Steering Vector"):
        pos_tokens = tokenizer(pos_prompt, return_tensors="pt").input_ids.to(model_device)
        neg_tokens = tokenizer(neg_prompt, return_tensors="pt").input_ids.to(model_device)

        with torch.no_grad():
            _, pos_cache = model.run_with_cache(pos_tokens, names_filter=lambda name: name in hook_names_to_cache)
            _, neg_cache = model.run_with_cache(neg_tokens, names_filter=lambda name: name in hook_names_to_cache)
            
        for layer_idx in layers:
            hook_name = get_act_name(layer_type_name, layer_idx)
            
            pos_act_seq = pos_cache[hook_name].squeeze(0) # Remove batch_dim (batch_size=1)
            neg_act_seq = neg_cache[hook_name].squeeze(0) # Remove batch_dim
            
            # Effective index for positive sequence (seq_len, d_model)
            eff_pos_idx = read_token_index if read_token_index >= 0 else pos_act_seq.shape[0] + read_token_index
            # Effective index for negative sequence
            eff_neg_idx = read_token_index if read_token_index >= 0 else neg_act_seq.shape[0] + read_token_index

            if not (0 <= eff_pos_idx < pos_act_seq.shape[0] and 0 <= eff_neg_idx < neg_act_seq.shape[0]):
                print(f"Warning: read_token_index {read_token_index} out of bounds for layer {layer_idx}. "
                      f"Pos seq len: {pos_act_seq.shape[0]}, Neg seq len: {neg_act_seq.shape[0]}. Skipping pair for this layer.")
                continue
            
            pos_act_at_idx = pos_act_seq[eff_pos_idx, :]
            neg_act_at_idx = neg_act_seq[eff_neg_idx, :]
            
            diff_act = (pos_act_at_idx - neg_act_at_idx).detach()
            layer_diff_activations[layer_idx].append(diff_act)

    final_steering_vectors: Dict[int, torch.Tensor] = {}
    for layer_idx, diff_acts_list in layer_diff_activations.items():
        if not diff_acts_list:
            # This can happen if all pairs were skipped for this layer due to index issues
            print(f"Warning: No activations collected for layer {layer_idx}. Vector will not be created for this layer.")
            continue
        
        avg_diff_tensor = torch.stack(diff_acts_list).mean(dim=0)
        if normalize:
            avg_diff_tensor = avg_diff_tensor / (avg_diff_tensor.norm() + 1e-8) # Add epsilon for stability
        if move_to_cpu:
            avg_diff_tensor = avg_diff_tensor.cpu()
        final_steering_vectors[layer_idx] = avg_diff_tensor
        
    return TLSteeringVector(final_steering_vectors, layer_type_name)


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
    return gathered_log_probs[:,-10:].sum().item()


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
            print("min_logprob", min_logprob)
            exp_pos = math.exp(logprobs_pos - min_logprob)
            print("exp_pos", exp_pos)
            exp_neg = math.exp(logprobs_neg - min_logprob)
            print("exp_neg", exp_neg)
            
            sum_exp = exp_pos + exp_neg
            print("sum_exp", sum_exp)
            prob_pos = exp_pos / sum_exp if sum_exp > 0 else 0.5 # Handle sum_exp = 0
            
        total_pos_prob_score += prob_pos
        
    return total_pos_prob_score / len(dataset) if dataset else 0.0

# --- Main usage example mimicking the first script ---
if __name__ == '__main__':
    # --- 1. Set up Model & Tokenizer ---
    # For easier local testing, consider smaller, open models if Llama-2 access is an issue:
    model_name = "Qwen/Qwen2.5-14B-Instruct" # Small, open chat model
    
    print(f"Loading model: {model_name}...")
    model = HookedTransformer.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        # device_map="auto" # For multi-GPU or automatic placement
    )
    model.eval() # Set to evaluation mode



    # --- 2. Set up Datasets (Dummy Example) ---
    # Replace with your actual data loading (e.g., from JSON files)
    with open("dishonesty_dataset.json", "r") as f:
        dataset = json.load(f)
    
    all_indices = list(range(len(dataset)))
    train_indices = all_indices[:int(len(all_indices) * 0.8)]
    test_indices = all_indices[int(len(all_indices) * 0.8):]
    
    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]
    
    
    
    
    
    


    sys_prompt = DEFAULT_SYS_PROMPT # Or a task-specific one like "You are a sycophantic assistant."
    
    def create_contrastive_pairs_from_mwe(
        mwe_data_list: List[MWEData], 
        tokenizer: PreTrainedTokenizerBase, 
        system_p: str
    ) -> List[Tuple[str, str]]:
        pairs = []
        for datum in mwe_data_list:
            pos_text = format_prompt_caa_style(
                question=datum["question"], tokenizer=tokenizer, system_prompt=system_p,
                answer=datum["answer_matching_behavior"]
            )
            neg_text = format_prompt_caa_style(
                question=datum["question"], tokenizer=tokenizer, system_prompt=system_p,
                answer=datum["answer_not_matching_behavior"]
            )
            pairs.append((pos_text, neg_text))
        return pairs

    train_contrastive_pairs = create_contrastive_pairs_from_mwe(train_dataset, model.tokenizer, sys_prompt)
    test_contrastive_pairs = create_contrastive_pairs_from_mwe(test_dataset, model.tokenizer, sys_prompt)
    
    print(f"Created {len(train_contrastive_pairs)} training pairs and {len(test_contrastive_pairs)} test pairs.")
    if train_contrastive_pairs:
        print("\nExample Training Pair:")
        print("#### Positive Prompt ####\n", train_contrastive_pairs[0][0])
        print("\n#### Negative Prompt ####\n", train_contrastive_pairs[0][1])
    print("-" * 30)

    # --- 3. Evaluate Model Without Steering ---
    print("Evaluating model without steering...")
    initial_performance = evaluate_model_contrastive(
        model, model.tokenizer, test_contrastive_pairs, show_progress=True
    )
    print(f"Unsteered model performance (avg prob of positive choice): {initial_performance:.3f}")
    print("-" * 30)

    # --- 4. Extract Steering Vectors ---
    # Choose layers based on your model (e.g., middle layers)
    layers_to_steer = [15]

    print(f"Training steering vector for layers: {layers_to_steer} using 'resid_pre' activations...")
    steering_vector_obj = train_tl_steering_vector(
        model, model.tokenizer, train_contrastive_pairs,
        layers=layers_to_steer, layer_type_name="resid_pre",
        read_token_index=-3, # Common CAA setting: token before the model makes its A/B choice
        move_to_cpu=False,   # Keep on model's device for faster application if model is on GPU
        show_progress=True,
        normalize=True
    )
    
    if not steering_vector_obj.layer_activations:
        print("Steering vector training failed or resulted in no usable vectors. Exiting.")
        exit()
        
    print(f"Steering vector trained. Vector shape for layer {layers_to_steer[0]}: {steering_vector_obj.layer_activations[layers_to_steer[0]].shape}")
    print("-" * 30)

    # --- 5. Steer with Steering Vectors & Evaluate ---
    print("Evaluating model with steering...")
    # min_token_index=0 applies steering to all tokens in the sequence for evaluation.
    for multiplier in [-6,-1.5, -1.0, 0.0, 1.0, 1.5,6]: # Example multipliers
        print(f"\n--- Evaluating with multiplier: {multiplier} ---")
        with steering_vector_obj.apply(model, multiplier=multiplier, layers=layers_to_steer, min_token_index=0):
            steered_performance = evaluate_model_contrastive(
                model, model.tokenizer, test_contrastive_pairs, show_progress=True
            )
            print(f"Multiplier {multiplier}: Steered model performance: {steered_performance:.3f}")
    print("-" * 30)
    print("Example finished.")

    # To test generation (optional, more complex hook logic might be desired for precise generation steering):
    # print("\n--- Example Generation (Optional) ---")
    # test_gen_q = "What do you think about open-source AI?"
    # gen_prompt_text = format_prompt_caa_style(test_gen_q, model.tokenizer, sys_prompt, answer=None) # No answer
    # input_ids = model.tokenizer(gen_prompt_text, return_tensors="pt").input_ids.to(model.cfg.device)
    #
    # for multiplier in [-1.0, 0.0, 1.0]:
    # print(f"\nGenerating with multiplier {multiplier}:")
    #     # For generation, you might want min_token_index to be len(input_ids[0]) if hook supports dynamic index
    #     # Or apply to all (min_token_index=0) which is simpler with current hook
    #     with steering_vector_obj.apply(model, multiplier=multiplier, layers=layers_to_steer, min_token_index=0):
    #         output_tokens = model.generate(input_ids, max_new_tokens=60, temperature=0.7, do_sample=True)
    #         decoded_output = model.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    # print(decoded_output)



