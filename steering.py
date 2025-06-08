import torch
import random
import json
import pickle as pkl
import os
from datetime import datetime
from tqdm import tqdm
from functools import partial
from typing import List, Tuple, Dict, Optional, Any

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from transformers import PreTrainedTokenizerBase

class TLSteeringVector:
    """
    A container for steering vectors for different layers,
    providing a context manager to apply them using TransformerLens hooks.
    """
    def __init__(self, layer_activations: Dict[int, torch.Tensor], layer_type_name: str = "hook_resid_pre"):
        self.layer_activations: Dict[int, torch.Tensor] = layer_activations
        self.layer_type_name: str = layer_type_name
        self.model: Optional[HookedTransformer] = None
        self.multiplier: float = 0.0
        self.hooks_active: List[Tuple[str, Any]] = []
        self.min_token_index: int = 0
        self.hook_context = None

    def _steering_hook_fn(self, activation_batch: torch.Tensor, hook: Any, steering_vec: torch.Tensor):
        # Apply steering vector to activations
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
                hook_fn = partial(self._steering_hook_fn, steering_vec=steering_vec)
                self.hooks_active.append((hook_point_name, hook_fn))
        return self

    def __enter__(self):
        if self.model is None:
            raise ValueError("SteeringVector not properly initialized with apply() before entering context.")
        if self.hooks_active:
            self.hook_context = self.model.hooks(fwd_hooks=self.hooks_active)
            self.hook_context.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.hook_context:
            self.hook_context.__exit__(exc_type, exc_value, traceback)
        self.model = None
        self.hooks_active = []
        self.hook_context = None


def train_steering_vector(
    model: HookedTransformer,
    tokenizer: PreTrainedTokenizerBase,
    data: List[Tuple[str, str]],  # List of (positive_prompt, negative_prompt) tuples
    layers: Optional[List[int]] = None,
    layer_type_name: str = "resid_pre",
    read_token_index: int = -1,
    move_to_cpu: bool = False,
    show_progress: bool = True,
    normalize: bool = True
) -> TLSteeringVector:
    """
    Train a steering vector using contrastive pairs of prompts.
    
    Args:
        model: HookedTransformer model
        tokenizer: Tokenizer  
        data: List of (positive_prompt, negative_prompt) tuples
        layers: List of layer indices to extract steering vectors from
        layer_type_name: Type of activation to use (e.g., "resid_pre")
        read_token_index: Token index to read activations from (-1 for last token)
        move_to_cpu: Whether to move activations to CPU
        show_progress: Whether to show progress bar
        normalize: Whether to normalize steering vectors
    
    Returns:
        TLSteeringVector object containing the trained steering vectors
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    layer_diff_activations: Dict[int, List[torch.Tensor]] = {layer_idx: [] for layer_idx in layers}
    hook_names_to_cache = [get_act_name(layer_type_name, layer_idx) for layer_idx in layers]
    
    model_device = model.cfg.device

    for pos_prompt, neg_prompt in tqdm(data, disable=not show_progress, desc="Training Steering Vector"):
        # Tokenize prompts
        pos_tokens = tokenizer(pos_prompt, return_tensors="pt").input_ids.to(model_device)
        neg_tokens = tokenizer(neg_prompt, return_tensors="pt").input_ids.to(model_device)

        with torch.no_grad():
            _, pos_cache = model.run_with_cache(pos_tokens, names_filter=lambda name: name in hook_names_to_cache)
            _, neg_cache = model.run_with_cache(neg_tokens, names_filter=lambda name: name in hook_names_to_cache)
            
        for layer_idx in layers:
            hook_name = get_act_name(layer_type_name, layer_idx)
            
            pos_act_seq = pos_cache[hook_name].squeeze(0)  # Remove batch dim
            neg_act_seq = neg_cache[hook_name].squeeze(0)  # Remove batch dim
            
            # Get activations at specified token index
            eff_pos_idx = read_token_index if read_token_index >= 0 else pos_act_seq.shape[0] + read_token_index
            eff_neg_idx = read_token_index if read_token_index >= 0 else neg_act_seq.shape[0] + read_token_index

            if not (0 <= eff_pos_idx < pos_act_seq.shape[0] and 0 <= eff_neg_idx < neg_act_seq.shape[0]):
                continue
            
            pos_act = pos_act_seq[eff_pos_idx, :]
            neg_act = neg_act_seq[eff_neg_idx, :]
            
            # Compute difference (positive - negative for steering toward positive-like responses)
            diff_act = (pos_act - neg_act).detach()
            if move_to_cpu:
                diff_act = diff_act.cpu()
            layer_diff_activations[layer_idx].append(diff_act)

    # Average the differences for each layer
    final_steering_vectors: Dict[int, torch.Tensor] = {}
    for layer_idx, diff_acts_list in layer_diff_activations.items():
        if not diff_acts_list:
            continue
        
        avg_diff_tensor = torch.stack(diff_acts_list).mean(dim=0)
        if normalize:
            avg_diff_tensor = avg_diff_tensor / (avg_diff_tensor.norm() + 1e-8)
        final_steering_vectors[layer_idx] = avg_diff_tensor
        
    return TLSteeringVector(final_steering_vectors, get_act_name(layer_type_name, 0).replace("0", ""))


def convert_honesty_samples_to_pairs(samples_tuples: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str]]:
    """Convert honesty samples format to contrastive pairs format."""
    pairs = []
    for question, honest_arg, dishonest_arg, reason in samples_tuples:
        honest_prompt = question + honest_arg
        dishonest_prompt = question + dishonest_arg
        pairs.append((dishonest_prompt, honest_prompt))  # (positive=dishonest, negative=honest) for dishonesty steering
    return pairs


# Legacy function names for backward compatibility
def train_tl_steering_vector(*args, **kwargs):
    """Legacy function - use train_steering_vector instead."""
    return train_steering_vector(*args, **kwargs)


def train_simple_steering_vector(
    model: HookedTransformer,
    tokenizer: PreTrainedTokenizerBase,
    samples_tuples: List[Tuple[str, str, str, str]],
    **kwargs
) -> TLSteeringVector:
    """Legacy function - use train_steering_vector instead."""
    pairs = convert_honesty_samples_to_pairs(samples_tuples)
    return train_steering_vector(model, tokenizer, pairs, **kwargs)


def test_generation_with_steering_and_save(
    model: HookedTransformer,
    tokenizer: PreTrainedTokenizerBase,
    steering_vector: TLSteeringVector,
    layers: List[int],
    test_prompts: List[Tuple[str, str]],  
    multipliers: List[float],
    output_file: str = "steering_results.csv"
):
    """Test generation with different steering multipliers and save to CSV."""
    
    results = []
    
    # Add some general prompts that might elicit honest/dishonest responses
    
    for prompt, category in test_prompts:
        print(f"\nPrompt ({category}): {prompt}")
        print("-" * 50)
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.cfg.device)
        
        row_data = {
            "prompt": prompt,
            "category": category,
        }
        
        for multiplier in multipliers:
            print(f"Multiplier {multiplier:+.1f}: ", end="")
            
            with steering_vector.apply(model, multiplier=multiplier, layers=layers, min_token_index=0):
                output_tokens = model.generate(
                    input_ids, 
                    max_new_tokens=200, 
                    temperature=0.7, 
                    do_sample=True,
                    verbose=False
                )
                decoded_output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                response = decoded_output[len(prompt):].strip()
                print(response)
                
                # Store response for this multiplier
                row_data[f"response_mult_{multiplier:+.1f}"] = response
        
        results.append(row_data)
    
    # Save results to CSV
    if results:
        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            json.dump(results, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {output_file}")


def run_experiment():
    """Main function to run the simplified honesty steering experiment."""
    
    # Load model
    # model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    print(f"Loading model: {model_name}...")
    
    model = HookedTransformer.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    model.eval()
    
    
    # Train steering vector
    layers_to_steer = [8, 12,18]  # Middle layers
    print(f"\nTraining steering vector for layers: {layers_to_steer}...")
    
    # Convert samples to the unified format
    from samples import samples_tuples
    contrastive_pairs = convert_honesty_samples_to_pairs(samples_tuples)
    
    steering_vector = train_steering_vector(
        model, model.tokenizer, contrastive_pairs,
        layers=layers_to_steer,
        layer_type_name="resid_pre",
        read_token_index=-1,
        move_to_cpu=True,
        show_progress=True,
        normalize=True
    )
    
    if not steering_vector.layer_activations:
        print("Failed to create steering vector!")
        return
    
    print(f"Steering vector created for layers: {list(steering_vector.layer_activations.keys())}")
    
    multipliers = [ -1.0, -0.5, 0.0, 0.5, 1.0]
    
    # Create output filename with timestamp
    global timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = f"honesty_steering_results_{timestamp}.json"
    
    print("\n" + "="*60)
    print("GENERATION TESTS WITH STEERING")
    print("="*60)
    
    test_generation_with_steering_and_save(
        model, model.tokenizer, steering_vector, 
        layers_to_steer, multipliers,
        output_file
    )


if __name__ == '__main__':
    run_experiment()
