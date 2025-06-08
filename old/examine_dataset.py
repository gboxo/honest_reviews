# %%
import json

# %%
all_prompts = []
with open("final_prompts.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        all_prompts.append(data["full_prompt"])

# %%

# %%
"qwen/qwen-2.5-coder-32b-instruct"