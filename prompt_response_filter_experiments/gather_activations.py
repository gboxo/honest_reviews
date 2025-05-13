import random
import json
import os
from transformer_lens import HookedTransformer
import torch
import numpy as np
import pandas as pd
from template import introduction, final_instructions
from tqdm import tqdm



if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-14B-Instruct", torch_dtype=torch.float16)
    promtps = "final_prompts.jsonl"
    arguments = "final_prompts_arguments.json"
    all_prompts = []
    with open(promtps, "r") as f:
        for line in f:
            data = json.loads(line)
            all_prompts.append((data["variant"],data["full_prompt"]))
     
    
    
    
    says_assured = []
    with open(arguments, "r") as f:
        arguments = json.load(f)
    
    all_arguments = []
    for key, value in arguments.items():
        all_arguments.append((value["prompt_index"],value["argument"]))
    
    says_assured = []
    for i in range(len(all_arguments)):
        if "assured" in all_arguments[i][1]:
            says_assured.append(i)
    
    entries_to_remove = []
    for idx in says_assured:
        entries_to_remove.extend([idx, idx+1])


    
    
    
    
    
    
        
        
    # Format the prompts with chat template
    all_tokens = []
    for prompt_tuple, argument_tuple in zip(all_prompts, all_arguments):

        prompt = prompt_tuple[1]
        argument = argument_tuple[1]




        messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": argument}
        ]
        text = model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        tokenized_prompt = model.tokenizer([text], return_tensors="pt").to("cuda")
        all_tokens.append(tokenized_prompt["input_ids"])
    
    filtered_all_tokens = [all_tokens[i] for i in range(len(all_tokens)) if i not in entries_to_remove]
    all_tokens = filtered_all_tokens
    print(len(all_tokens))
    
    # Get the activations

    all_final_activations = []

    names_filter = lambda name: "hook_resid_pre" in name


    all_final_activations = []
    for tokens in tqdm(all_tokens):
        with torch.no_grad():
            logits,cache = model.run_with_cache(tokens, names_filter=names_filter)
            all_activations  = [cache[f"blocks.{i}.hook_resid_pre"][:,-1] for i in range(len(cache))]
            all_activations = torch.stack(all_activations)[:,0].cpu()
            all_final_activations.append(all_activations)

    all_final_activations = torch.stack(all_final_activations)
    os.makedirs("activations", exist_ok=True)
    torch.save(all_final_activations, "activations/final_activations_prompts.pt")


    


    