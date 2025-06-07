"""
Simplified bilingual steering experiment with short phrases in English and Spanish
"""

import torch
import random
import csv
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


def create_expanded_bilingual_phrases():
    """Create expanded bilingual phrases for steering."""
    phrases = [
        # Simple statements in English and Spanish
        ("I love to read books", "Me encanta leer libros"),
        ("The weather is beautiful today", "El clima está hermoso hoy"),
        ("My favorite color is blue", "Mi color favorito es azul"),
        ("I enjoy cooking pasta", "Me gusta cocinar pasta"),
        ("The cat sleeps on the couch", "El gato duerme en el sofá"),
        
        # More examples
        ("Let's go to the beach", "Vamos a la playa"),
        ("The coffee is very hot", "El café está muy caliente"),
        ("I live in the city", "Vivo en la ciudad"),
        ("The flowers are yellow", "Las flores son amarillas"),
        ("She plays the piano well", "Ella toca bien el piano"),
        
        # Additional expanded examples
        ("The sun shines brightly", "El sol brilla intensamente"),
        ("I walk in the park", "Camino en el parque"),
        ("The dog runs fast", "El perro corre rápido"),
        ("Music makes me happy", "La música me hace feliz"),
        ("The mountain is very tall", "La montaña es muy alta"),
        
        ("Children play in the garden", "Los niños juegan en el jardín"),
        ("The ocean is deep blue", "El océano es azul profundo"),
        ("I drink water every day", "Bebo agua todos los días"),
        ("The bird sings beautifully", "El pájaro canta hermosamente"),
        ("My family is important", "Mi familia es importante"),
        
        ("The moon appears at night", "La luna aparece por la noche"),
        ("I study mathematics daily", "Estudio matemáticas diariamente"),
        ("The tree has green leaves", "El árbol tiene hojas verdes"),
        ("We eat dinner together", "Cenamos juntos"),
        ("The car drives smoothly", "El carro maneja suavemente"),
        
        ("Books contain knowledge", "Los libros contienen conocimiento"),
        ("The river flows gently", "El río fluye suavemente"),
        ("I write letters carefully", "Escribo cartas cuidadosamente"),
        ("The stars twinkle above", "Las estrellas brillan arriba"),
        ("Friends share good times", "Los amigos comparten buenos momentos"),
        
        ("The kitchen smells wonderful", "La cocina huele maravilloso"),
        ("I learn something new", "Aprendo algo nuevo"),
        ("The garden grows vegetables", "El jardín cultiva vegetales"),
        ("We travel to new places", "Viajamos a lugares nuevos"),
        ("The clock shows the time", "El reloj muestra la hora"),
        
        ("Art expresses creativity", "El arte expresa creatividad"),
        ("The wind blows softly", "El viento sopla suavemente"),
        ("I help my neighbors", "Ayudo a mis vecinos"),
        ("The fire warms the room", "El fuego calienta la habitación"),
        ("Dreams inspire hope", "Los sueños inspiran esperanza"),
    ]
    return phrases


def create_expanded_test_prompts():
    """Create expanded test prompts in both languages."""
    prompts = [
        # Basic questions
        ("What's your favorite color?", "en"),
        ("¿Cuál es tu color favorito?", "es"),
        ("Tell me about the weather today.", "en"),
        ("Háblame del clima hoy.", "es"),
        ("How do you make pizza?", "en"),
        ("¿Cómo se hace la pizza?", "es"),
        
        # Expanded questions
        ("What do you like to do for fun?", "en"),
        ("¿Qué te gusta hacer para divertirte?", "es"),
        ("Describe your perfect day.", "en"),
        ("Describe tu día perfecto.", "es"),
        ("What's your favorite book?", "en"),
        ("¿Cuál es tu libro favorito?", "es"),
        
        ("How do you feel about music?", "en"),
        ("¿Cómo te sientes sobre la música?", "es"),
        ("What's the best place to visit?", "en"),
        ("¿Cuál es el mejor lugar para visitar?", "es"),
        ("Tell me about your family.", "en"),
        ("Háblame de tu familia.", "es"),
        
        ("What makes you happy?", "en"),
        ("¿Qué te hace feliz?", "es"),
        ("How do you spend your weekends?", "en"),
        ("¿Cómo pasas tus fines de semana?", "es"),
        ("What's your favorite food?", "en"),
        ("¿Cuál es tu comida favorita?", "es"),
        
        ("Describe the ocean.", "en"),
        ("Describe el océano.", "es"),
        ("What do you think about nature?", "en"),
        ("¿Qué piensas sobre la naturaleza?", "es"),
        ("How do you learn new things?", "en"),
        ("¿Cómo aprendes cosas nuevas?", "es"),
        
        ("What's your dream vacation?", "en"),
        ("¿Cuáles son tus vacaciones soñadas?", "es"),
        ("Tell me about friendship.", "en"),
        ("Háblame sobre la amistad.", "es"),
        ("What inspires you most?", "en"),
        ("¿Qué te inspira más?", "es"),
    ]
    return prompts


def train_simple_steering_vector(
    model: HookedTransformer,
    tokenizer: PreTrainedTokenizerBase,
    phrase_pairs: List[Tuple[str, str]],  # List of (english_phrase, spanish_phrase)
    layers: Optional[List[int]] = None,
    layer_type_name: str = "resid_pre",
    read_token_index: int = -1,
    show_progress: bool = True,
    normalize: bool = True
) -> TLSteeringVector:
    """
    Train a steering vector using simple phrase pairs.
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    layer_diff_activations: Dict[int, List[torch.Tensor]] = {layer_idx: [] for layer_idx in layers}
    hook_names_to_cache = [get_act_name(layer_type_name, layer_idx) for layer_idx in layers]
    
    model_device = model.cfg.device

    for english_phrase, spanish_phrase in tqdm(phrase_pairs, disable=not show_progress, desc="Training Steering Vector"):
        # Tokenize phrases
        english_tokens = tokenizer(english_phrase, return_tensors="pt").input_ids.to(model_device)
        spanish_tokens = tokenizer(spanish_phrase, return_tensors="pt").input_ids.to(model_device)

        with torch.no_grad():
            _, english_cache = model.run_with_cache(english_tokens, names_filter=lambda name: name in hook_names_to_cache)
            _, spanish_cache = model.run_with_cache(spanish_tokens, names_filter=lambda name: name in hook_names_to_cache)
            
        for layer_idx in layers:
            hook_name = get_act_name(layer_type_name, layer_idx)
            
            english_act_seq = english_cache[hook_name].squeeze(0)  # Remove batch dim
            spanish_act_seq = spanish_cache[hook_name].squeeze(0)  # Remove batch dim
            
            # Get activations at specified token index
            eff_english_idx = read_token_index if read_token_index >= 0 else english_act_seq.shape[0] + read_token_index
            eff_spanish_idx = read_token_index if read_token_index >= 0 else spanish_act_seq.shape[0] + read_token_index

            if not (0 <= eff_english_idx < english_act_seq.shape[0] and 0 <= eff_spanish_idx < spanish_act_seq.shape[0]):
                continue
            
            english_act = english_act_seq[eff_english_idx, :]
            spanish_act = spanish_act_seq[eff_spanish_idx, :]
            
            # Compute difference (spanish - english for steering toward Spanish-like responses)
            diff_act = (spanish_act - english_act).detach()
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
        
    return TLSteeringVector(final_steering_vectors, layer_type_name)


def test_generation_with_steering_and_save(
    model: HookedTransformer,
    tokenizer: PreTrainedTokenizerBase,
    steering_vector: TLSteeringVector,
    test_prompts: List[Tuple[str, str]],  # List of (prompt, language)
    layers: List[int],
    multipliers: List[float],
    output_file: str = "steering_results.csv"
):
    """Test generation with different steering multipliers and save to CSV."""
    
    results = []
    
    for prompt, language in test_prompts:
        print(f"\nPrompt ({language}): {prompt}")
        print("-" * 50)
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.cfg.device)
        
        row_data = {
            "prompt": prompt,
            "language": language,
        }
        
        for multiplier in multipliers:
            print(f"Multiplier {multiplier:+.1f}: ", end="")
            
            with steering_vector.apply(model, multiplier=multiplier, layers=layers, min_token_index=0):
                output_tokens = model.generate(
                    input_ids, 
                    max_new_tokens=40, 
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
        fieldnames = ["prompt", "language"] + [f"response_mult_{mult:+.1f}" for mult in multipliers]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults saved to {output_file}")


def run_simple_bilingual_experiment():
    """Main function to run the simplified bilingual steering experiment."""
    
    # Load model
    # model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Loading model: {model_name}...")
    
    model = HookedTransformer.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    model.eval()
    
    # Create expanded phrase pairs
    print("Creating expanded bilingual phrase pairs...")
    phrase_pairs = create_expanded_bilingual_phrases()
    
    print(f"Total phrase pairs: {len(phrase_pairs)}")
    print("Example phrase pairs:")
    for i, (english, spanish) in enumerate(phrase_pairs[:5]):
        print(f"{i+1}. English: '{english}' | Spanish: '{spanish}'")
    
    # Train steering vector
    layers_to_steer = [8, 12]  # Middle layers
    print(f"\nTraining steering vector for layers: {layers_to_steer}...")
    
    steering_vector = train_simple_steering_vector(
        model, model.tokenizer, phrase_pairs,
        layers=layers_to_steer,
        layer_type_name="resid_pre",
        read_token_index=-1,
        show_progress=True,
        normalize=True
    )
    
    if not steering_vector.layer_activations:
        print("Failed to create steering vector!")
        return
    
    print(f"Steering vector created for layers: {list(steering_vector.layer_activations.keys())}")
    
    # Create expanded test prompts
    test_prompts = create_expanded_test_prompts()
    print(f"\nTotal test prompts: {len(test_prompts)}")
    
    multipliers = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"bilingual_steering_results_{timestamp}.csv"
    
    print("\n" + "="*60)
    print("GENERATION TESTS WITH STEERING")
    print("="*60)
    
    test_generation_with_steering_and_save(
        model, model.tokenizer, steering_vector, 
        test_prompts, layers_to_steer, multipliers,
        output_file
    )
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print("="*60)
    print("Negative multipliers should steer toward English-like responses")
    print("Positive multipliers should steer toward Spanish-like responses")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    run_simple_bilingual_experiment()