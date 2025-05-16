import json
import random
import argparse
from template import introduction, dishonest_suffix, final_instructions, category_and_type

def process_spec_sheet(spec_sheet):
    """
    Scan through the specification sheet and return a tuple of:
    - selected_feature: one key (whose value is greater than 7)
    - modified_spec: a copy where that feature is set to 3.
    Excludes non-rating keys like name, company_name, year, or year_of_launch.
    """
    ignore_keys = {"name", "company_name", "year", "year_of_launch"}
    eligible_features = [
        key for key, value in spec_sheet.items()
        if key not in ignore_keys and isinstance(value, int) and value > 7
    ]
    if not eligible_features:
        return None, spec_sheet
    selected_feature = random.choice(eligible_features)
    modified_spec = spec_sheet.copy()
    modified_spec[selected_feature] = 3
    return selected_feature, modified_spec

def get_product_category(product_type):
    """
    Using the mapping from category_and_type, find the category corresponding
    to the given product type.
    """
    for category, types in category_and_type.items():
        if product_type in types:
            return category
    return None

def create_prompts(input_filepath, output_filepath):
    """
    Reads the specification sheet jsonl file, processes each record by modifying
    one of its feature ratings, fills all template parts, and writes out TWO
    jsonl records per input:
      - One version includes the dishonest suffix.
      - The other version does not include the dishonest suffix.
    """
    with open(input_filepath, "r") as infile, open(output_filepath, "w") as outfile:
        for line in infile:
            if not line.strip():
                continue  # skip empty lines
            data = json.loads(line)
            product_type = data.get("product_type")
            spec_sheet = data.get("specification_sheet", {})
            product_category = get_product_category(product_type)
            if not product_category:
                print(f"Skipping product_type {product_type}: category not found.")
                continue

            selected_feature, modified_spec = process_spec_sheet(spec_sheet)
            if not selected_feature:
                print(f"Skipping product_type {product_type}: no eligible feature found.")
                continue

            # For readability, replace underscores with spaces.
            readable_feature = selected_feature.replace("_", " ")

            # Fill in the template parts
            filled_introduction = introduction.format(
                product_category=product_category, product_type=product_type
            )
            filled_dishonest_suffix = dishonest_suffix.format(
                feature=readable_feature, product_type=product_type
            )
            instructions = final_instructions

            # Prepare the specification sheet as a JSON code block.
            spec_sheet_json = json.dumps(modified_spec, indent=2)
            json_block = "```json\n" + spec_sheet_json + "\n```"

            # Create two versions:
            # 1. Version with the dishonest suffix.
            dishonest_prompt = (
                f"{filled_introduction}\n\n"
                f"{json_block}\n\n"
                f"{filled_dishonest_suffix}\n\n"
                f"{instructions}"
            )
            
            # 2. Version without the dishonest suffix.
            honest_prompt = (
                f"{filled_introduction}\n\n"
                f"{json_block}\n\n"
                f"{instructions}"
            )

            # Build output objects for each variant. We add a 'variant' key to distinguish them.
            output_obj_dishonest = {
                "product_type": product_type,
                "product_category": product_category,
                "selected_feature": selected_feature,
                "original_specification_sheet": spec_sheet,
                "modified_specification_sheet": modified_spec,
                "filled_introduction": filled_introduction,
                "filled_dishonest_suffix": filled_dishonest_suffix,
                "final_instructions": instructions,
                "full_prompt": dishonest_prompt,
                "variant": "dishonest"
            }
            
            output_obj_honest = {
                "product_type": product_type,
                "product_category": product_category,
                "selected_feature": selected_feature,
                "original_specification_sheet": spec_sheet,
                "modified_specification_sheet": modified_spec,
                "filled_introduction": filled_introduction,
                "final_instructions": instructions,
                "full_prompt": honest_prompt,
                "variant": "honest"
            }

            # Write both outputs to the file (each as a separate jsonl entry)
            outfile.write(json.dumps(output_obj_dishonest) + "\n")
            outfile.write(json.dumps(output_obj_honest) + "\n")

def main():
    parser = argparse.ArgumentParser(description='Create prompts from specification sheets')
    parser.add_argument('--input', '-i', help='Input JSONL file containing specification sheets')
    parser.add_argument('--output', '-o', default="final_new_spec_sheet.jsonl", help='Output JSONL file for generated prompts')
    args = parser.parse_args()
    
    create_prompts(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
