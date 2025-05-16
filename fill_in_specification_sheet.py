import json
import re
import openai
import argparse
from pathlib import Path



from template import category_and_type

"""
Send request to generate synthetic examples for a specification sheet.

"""



def validate_evaluation(evaluation):
    """
    The returned evaluation object should have the propoerties:
    - name: str
    - model: str
    - brand: str
    - year: int
    The other features are optional, but if they are included, they should be integers from 0 to 10.
    """

    if not isinstance(evaluation, dict):
        return False
    if not all(key in evaluation for key in ["name", "model", "brand", "year"]):
        return False
    for key in evaluation:
        if key not in ["name", "model", "brand", "year"]:
            if not isinstance(evaluation[key], int):
                return False
    return True





# Prompt for generating a specification sheet



prompt_template = """
You are tasked with generating synthetic specification sheets for products, as would be used in a product review process. You will be provided with the product type, its category, and a JSON skeleton listing various features that are rated on a scale from 0 to 10.

Your task is to fill in the specification sheet for a **{product_type}** in the **{category}** category. The completed specification sheet should be in valid JSON format, with appropriate data types for each feature.

**Specification Sheet:**

{specification_sheet}



---

**Example:**

- **Category:** Consumer Electronics  
- **Product Type:** Smartphone

**Specification Sheet:**

```json
{{
  "name": "Zephyr X1",
  "model": "ZX1-2025",
  "brand": "NovaTech",
  "year": 2025,
  "operating_system": "NovaOS 3.2",
  "battery_life": 9,
  "camera_quality": 3,
  "processor_performance": 9,
  "screen_clarity": 8,
  "build_quality": 9,
  "audio_quality": 8,
  "charging_speed": 9,
  "software_experience": 8,
  "multitasking_performance": 9,
  "front_camera_quality": 4,
  "video_recording_quality": 3,
  "fingerprint_sensor_accuracy": 9
}}
```

---

**Instructions:**

- Ensure the JSON is valid and properly formatted.
- Include the product name, brand, and year of release.
- Do not use real brands or product names.
- Use plausible, varied ratings for each feature.
"""




# LLM calls with the prompt and the product type and category


def generate_specification_sheet(product_type, category, prompt_template, specification_sheet):
    model = "gpt-4.1-2025-04-14"

    system_prompt = "You are a helpful assistant that generates specification sheet for a product."

    
    
    
    prompt  = prompt_template.format(product_type=product_type, category=category, specification_sheet=specification_sheet)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1000
    )
    response_text = response.choices[0].message.content

    # Extract JSON from markdown if applicable.
    json_match = re.search(r'```json\s*({.*?})\s*```', response_text, re.DOTALL)
    json_str = json_match.group(1) if json_match else response_text.strip()


    # Convert string to dict
    print(json_str)
    spec_sheet = json.loads(json_str)
    #if not validate_specification_sheet(spec_sheet):
    #    raise ValueError("Invalid specification sheet format")
    return spec_sheet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fill in specification sheets with synthetic data')
    parser.add_argument('--input', '-i', 
                       default="specification_sheets.jsonl",
                       help='Input JSONL file containing specification sheets (default: specification_sheets.jsonl)')
    parser.add_argument('--output', '-o', 
                       default="specification_sheets_filled.jsonl",
                       help='Output JSONL file for filled specification sheets (default: specification_sheets_filled.jsonl)')
    parser.add_argument('--samples', '-n',
                       type=int,
                       default=4,
                       help='Number of samples to generate per specification sheet (default: 4)')
    args = parser.parse_args()

    dictionary_sepct_sheets = {}
    
    with open(args.input, "r") as f:
        for line in f:
            data = json.loads(line)
            category = data["category"]
            product_type = data["product_type"]
            specification_sheet = data["specification_sheet"]
            specification_sheet_str = json.dumps(specification_sheet).replace(",", ',\n')

            out = prompt_template.format(product_type=product_type, category=category, specification_sheet=specification_sheet_str)
            out_list = []
            for _ in range(args.samples):
                out = generate_specification_sheet(product_type, category, prompt_template, specification_sheet_str)
                out_list.append(out)

            dictionary_sepct_sheets[product_type] = out_list

    # Save the dictionary as a jsonl file
    with open(args.output, "w") as f:
        for product_type, specification_sheet_list in dictionary_sepct_sheets.items():
            for specification_sheet in specification_sheet_list:
                f.write(json.dumps({"product_type": product_type, "specification_sheet": specification_sheet}) + "\n")



