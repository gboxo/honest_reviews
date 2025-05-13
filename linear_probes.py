import torch
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score




def get_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred)
    }


def lr_probe(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegressionCV(cv=5, random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return get_metrics(y_test, y_pred)


def fit_lr_all_layers(activations):
    all_metrics = []

    for layer in range(activations.shape[1]):

        acts = activations[:, layer, :].cpu().numpy()
        y = np.array([0 if i % 2 == 0 else 1 for i in range(acts.shape[0])])
        metrics = lr_probe(acts, y)
        all_metrics.append(metrics)

    return all_metrics

if __name__ == "__main__":

    # Prompt and response experiments

    path = "prompt_response_experiments/activations/final_activations_prompts.pt"
    activations = torch.load(path)
    metrics = fit_lr_all_layers(activations)
    print(metrics)

    # Prompt and response experiments

    path = "honest_prompt_response_experiments/activations/final_activations_prompts.pt"
    activations = torch.load(path)
    metrics = fit_lr_all_layers(activations)
    print(metrics)

    # Prompt and response experiments

    path = "only_response_experiments/activations/final_activations_prompts.pt"
    activations = torch.load(path)
    metrics = fit_lr_all_layers(activations)
    print(metrics)