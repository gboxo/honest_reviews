import json
import os
from filelock import FileLock
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")


def load_prompts_no_id(file_path: str) -> List[Dict]:
    """
    Load prompts from a JSONL file.
    Each line is expected to be a JSON object.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_arguments_no_id(file_path: str) -> Dict:
    """
    Load previously saved arguments from a JSON file.
    Returns an empty dict if the file does not exist.
    """
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


class TextGenerator:
    def __init__(
        self,
        model_name: str = "qwen/qwen-2.5-coder-32b-instruct",
        temperature: float = 0.56,
        max_tokens: int = 512,
        openrouter_api_key: str = openrouter_api_key,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None
    ):
        """
        Initialize the text generator with model and configuration.

        Args:
            model_name (str): Name of the model to use from OpenRouter
            temperature (float): Sampling temperature for generation
            max_tokens (int): Maximum number of tokens to generate
            openrouter_api_key (str): API key for OpenRouter
            site_url (str): Optional. Site URL for rankings on OpenRouter
            site_name (str): Optional. Site title for rankings on OpenRouter
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json"
        }
        if site_url:
            self.headers["HTTP-Referer"] = site_url
        if site_name:
            self.headers["X-Title"] = site_name

    def save_arguments(self, file_path: str, arguments: Dict):
        """Save arguments to a JSON file with file locking."""
        lock = FileLock(f"{file_path}.lock")
        with lock:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(arguments, f, ensure_ascii=False, indent=4)

    def generate_single_text(self, prompt: str) -> str:
        """Generate text for a single prompt using the OpenRouter API."""
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "include_reasoning": True  # Include reasoning in the response
        }

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=self.headers,
                json=payload
            )
            if response.status_code == 200:
                result = response.json()
                text = result['choices'][0]['message']['content']
                return text
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_text_batch(self, prompts: List[str]) -> List[str]:
        """
        Generate text for a batch of prompts using multithreading,
        preserving order.
        """
        results = [None] * len(prompts)  # Pre-allocate list for ordered results
        future_to_index = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            for index, prompt in enumerate(prompts):
                future = executor.submit(self.generate_single_text, prompt)
                future_to_index[future] = index

            for future in tqdm(as_completed(future_to_index),
                               total=len(prompts),
                               desc="Generating Text"):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    print(f"Error processing prompt at index {index}: {e}")
                    results[index] = f"Error: Generation failed - {str(e)}"

        if None in results:
            print("Warning: Some prompts might not have generated results correctly.")
            results = [res if res is not None else "Error: Result missing" for res in results]

        return results

    def process_prompts(self, data: List[Dict], arguments: Dict, arguments_file: str, batch_size: int = 16) -> Dict:
        """
        Process prompts in batches, update arguments, and save incrementally.
        This version uses the "full_prompt" field from each JSON entry.
        """
        if arguments is None:
            arguments = {}

        unprocessed = []
        print("Identifying unprocessed prompts...")
        for idx, item in enumerate(data):
            if str(idx) in arguments:
                continue
            prompt = item.get("full_prompt", "").strip()
            if not prompt:
                arguments[str(idx)] = {
                    "error": "Empty prompt",
                    "dataset_index": idx,
                    "variant": item.get("variant", "unknown")
                }
                continue
            unprocessed.append((idx, item))
        print(f"Found {len(unprocessed)} unprocessed prompts.")

        total_batches = (len(unprocessed) + batch_size - 1) // batch_size

        for batch_num, batch_start in enumerate(range(0, len(unprocessed), batch_size)):
            batch = unprocessed[batch_start:batch_start + batch_size]
            if not batch:
                continue

            current_batch_num = batch_num + 1
            print(f"\n--- Processing Batch {current_batch_num}/{total_batches} ---")

            try:
                batch_indices = [idx for idx, _ in batch]
                # Use the "full_prompt" field instead of "prompt"
                batch_prompts = [item["full_prompt"].strip() for _, item in batch]

                results = self.generate_text_batch(batch_prompts)

                print(f"Updating arguments for batch {current_batch_num}...")
                for (idx, item), result in zip(batch, results):
                    arguments[str(idx)] = {
                        "argument": result,
                        "prompt_index": idx,
                        "product_type": item.get("product_type", ""),
                        "product_category": item.get("product_category", ""),
                        "selected_feature": item.get("selected_feature", ""),
                        "variant": item.get("variant", "honest"),
                        "generation_params": {
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens
                        }
                    }

                print(f"Saving progress after batch {current_batch_num} to {arguments_file}...")
                self.save_arguments(arguments_file, arguments)
                print(f"Successfully saved progress for batch {current_batch_num}.")

            except KeyboardInterrupt:
                print(f"\nKeyboardInterrupt detected during batch {current_batch_num}. Saving current progress...")
                self.save_arguments(arguments_file, arguments)
                print(f"Progress saved to {arguments_file}. Stopping processing.")
                raise KeyboardInterrupt
            except Exception as e:
                print(f"\nAn error occurred during batch {current_batch_num}: {e}. Saving current progress...")
                self.save_arguments(arguments_file, arguments)
                print(f"Progress saved to {arguments_file}. Stopping processing due to error.")
                raise e

        print("\n--- All batches processed ---")
        return arguments

    def run_generation(
        self,
        prompts_file: str,
        arguments_file: Optional[str] = None,
        batch_size: int = 16
    ):
        """
        Main method to run the generation process. Handles KeyboardInterrupt.
        If arguments_file is not provided, a default file will be created.
        
        Args:
            prompts_file (str): Path to the prompts JSONL file.
            arguments_file (str): Path to save/load arguments JSON.
            batch_size (int): Number of prompts to process in each batch.
        """
        if arguments_file is None:
            # Create a default arguments file name based on the prompts file name.
            base = prompts_file.replace(".jsonl", "")
            arguments_file = f"{base}_arguments.json"

        print(f"Loading prompts from: {prompts_file}")
        data = load_prompts_no_id(prompts_file)
        print(f"Loading existing arguments from: {arguments_file}")
        self.arguments = load_arguments_no_id(arguments_file)

        try:
            print("Starting prompt processing...")
            updated_arguments = self.process_prompts(
                data,
                self.arguments,
                arguments_file,
                batch_size
            )
            print("\nGeneration completed successfully.")

        except KeyboardInterrupt:
            print(f"\nGeneration interrupted by user. Final progress saved to {arguments_file}.")
        except Exception as e:
            print(f"\nAn unexpected error stopped the generation: {e}. Final progress saved to {arguments_file}.")


if __name__ == "__main__":
    # Instantiate the generator with the API key
    generator = TextGenerator(openrouter_api_key=openrouter_api_key)

    # Define file path for prompts file
    prompts_path = "final_prompts.jsonl"

    # Define the batch size for processing
    data_generation_batch_size = 10

    print("Starting generation process...")
    print(f"  Prompts file: {prompts_path}")
    print(f"  Batch size: {data_generation_batch_size}")

    generator.run_generation(
        prompts_file=prompts_path,
        batch_size=data_generation_batch_size
    )



