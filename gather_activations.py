import random
import json
from transformer_lens import HookedTransformer
import torch
import numpy as np
import pandas as pd
from template import introduction, final_instructions


random_phone_company_names = ["NovaTech", "TechNova"]
random_phone_models = ["X6","X9", "X10"]
years = [2024, 2025]
phone_features = ["battery_life", "camera_quality", "processor_performance", "screen_clarity", "build_quality", "audio_quality", "charging_speed", "software_experience", "multitasking_performance", "front_camera_quality", "video_recording_quality", "fingerprint_sensor_accuracy"]



phone_spec ={
  "name": "str",
  "model": "str",
  "brand": "str",
  "year": "int",
  "battery_life": "int",
  "camera_quality": "int",
  "processor_performance": "int",
  "screen_clarity": "int",
  "build_quality": "int",
  "audio_quality": "int",
  "charging_speed": "int",
  "software_experience": "int",
  "multitasking_performance": "int",
  "front_camera_quality": "int",
  "video_recording_quality": "int",
  "fingerprint_sensor_accuracy": "int"
}

spec_list = []
for brand in random_phone_company_names:
  for model in random_phone_models:
    for year in years:
        new_spec = phone_spec.copy()
        new_spec["name"] = f"{brand} {model}"
        new_spec["brand"] = brand
        new_spec["model"] = model
        new_spec["year"] = year
        # Good features 
        subset_phone_features_good = random.sample(phone_features, 3)
        for feature in subset_phone_features_good:
            new_spec[feature] = random.randint(7, 10)
        # Bad features
        for feature in phone_features:
            if feature not in subset_phone_features_good:
                new_spec[feature] = random.randint(1, 6)
        spec_list.append(new_spec)



if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

    
    if False:
        category = "Consumer Electronics"
        product_type = "Smartphone"
        introduction = introduction.format(product_category=category, product_type=product_type)
        
        spec_json = json.dumps(spec_list[-1], indent=2)
        
        final_prompt = f"{introduction}\n\n{spec_json}\n\n{final_instructions}"

        messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": final_prompt}
        ]


        text = model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = model.tokenizer([text], return_tensors="pt").to("cuda")

        
        
        with torch.no_grad():
            outputs = model.generate(
                model_inputs["input_ids"],
                max_new_tokens=1000,
                temperature=0.2,
                top_p=0.95,
                top_k=40,
                verbose = True
                    )
                
        
        
        out_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(out_text)


    