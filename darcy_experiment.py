import torch
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import re

from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
import json

class Lin_Model:
    # constructor + utils
    def __init__(self):
        self.model = LogisticRegressionCV(cv=5, random_state=42, max_iter=1000)  # cv = number of folds

    def get_metrics(self, y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred)
        }
    # fitting methods
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        return self.get_metrics(y_test, y_pred)

    def infer(self, X):
        return self.model.predict(X)
    
    def hello(self, X):
        print("Hello I am a linear model :)");


def get_activations(model, texts, tokenizer):
    """Get model activations for a list of texts"""
    all_activations = []
    
    for text in texts:
        # Format with chat template
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ]
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize and get activations
        tokens = tokenizer([formatted_text], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens["input_ids"])
            # Get activations from the last layer
            activations = cache[f"blocks.{model.cfg.n_layers-1}.hook_resid_pre"][0, -1].cpu()
            all_activations.append(activations)
    
    return torch.stack(all_activations)

def separate_question_and_choices(text):
    """Separate the question from the choices using regex"""
    # Split on "Choices:" and strip whitespace
    parts = re.split(r'\n\nChoices:\n ', text)
    if len(parts) != 2:
        raise ValueError(f"Could not separate question and choices in text: {text}")
    
    question = parts[0].strip()  # getting rid of nonsense
    choices = parts[1].strip()
    return question, choices

if __name__ == "__main__":
    # Load model and tokenizer
    model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-14B-Instruct", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

    # Load dataset
    path = "dishonesty_dataset.json"
    with open(path, "r") as f:
        data = json.load(f)

    # Separate questions and choices
    questions = []
    choices = []
    labels = []
    
    for item in data:
        # Label is 1 if answer_matching_behavior is (A), 0 if (B)
        labels.append(1 if item["answer_matching_behavior"].strip() == "(A)" else 0)
    
    labels = np.array(labels)

    # Get activations for questions and choices
    print("Getting question activations...")
    question_activations = get_activations(model, questions, tokenizer)
    print("Getting choice activations...")
    choice_activations = get_activations(model, choices, tokenizer)

    # Train linear probes
    print("Training question probe...")
    question_probe = Lin_Model()
    question_metrics = question_probe.fit(question_activations, labels)
    
    print("Training choice probe...")
    choice_probe = Lin_Model()
    choice_metrics = choice_probe.fit(choice_activations, labels)

    print("\nQuestion Probe Metrics:")
    print(question_metrics)
    print("\nChoice Probe Metrics:")
    print(choice_metrics)