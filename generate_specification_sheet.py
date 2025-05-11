import json
import re
import openai
from pathlib import Path



from template import category_and_type

"""
Send request to generate specification sheet for various devices.

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
You are tasked with generating specification sheet for a product.
This specification sheet will be used in the review process of a product. You will be given a product and the category it belongs to, and you will generate a json object that contain various features that can be rated from that product, from 0 to 10, in the review process.

You will need to generate a specification sheet for a {product_type} which belongs to the {category} category. Provide the specification sheet with json format, and the keys are the features that can be rated from 0 to 10. The values in the json object correspond to the data type of each feature (integer for the rating.)\nAlso in the specification sheet, include entries for the Name, the Company Name and the Year of lunch of the product.


---
**Example:**

- Category of the product: Consumer Electronics
- Type of the product: Smartphone

**Specification Sheet:**

```json
{
  name: str,
  model: str,
  brand: str,
  year: int,
  battery_life: int,
  camera_quality: int,
  processor_performance: int,
  screen_clarity: int,
  build_quality: int,
  audio_quality: int,
  charging_speed: int,
  software_experience: int,
  multitasking_performance: int,
  front_camera_quality: int,
  video_recording_quality: int,
  fingerprint_sensor_accuracy: int
}
```

---

**Instructions:**
1. The specification sheet should be in correctly formatted json.
2. There should be around 5 features in the specification sheet.
3. The features should be relevant to the product.
4. Make sure to include the Name, the Company Name and the Year of lunch of the product.
5. Make sure to include the data type of each feature (integer for the rating.)

""" 



# LLM calls with the prompt and the product type and category


def generate_specification_sheet(product_type, category, template):
    model = "gpt-4.1-2025-04-14"

    system_prompt = "You are a helpful assistant that generates specification sheet for a product."

    
    
    
    prompt  = template.format(product_type=product_type, category=category)

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
    dictionary_sepct_sheets = {}
    for category, category_type in category_and_type.items():
        dictionary_sepct_sheets[category] = {}
        for product_type, product_type_name in category_type.items():
            out = generate_specification_sheet(product_type, category, prompt_template)
            dictionary_sepct_sheets[category][product_type] = out


