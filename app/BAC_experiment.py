import pandas as pd
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from cross_validation import cross_validation_loop


EXPERIMENTS = [
    {
        "name": "1. Domyślne",
        "params": {},
    },
    {
        "name": "2. Balanced weight",
        "params": {
            "class_weight": "balanced",
        },
    },
    {
        "name": "3. SMOTE",
        "params": {
            "smote": True,
        },
    },
    {
        "name": "4. Balanced weight + SMOTE",
        "params": {
            "class_weight": "balanced",
            "smote": True,
        },
    },
    {
        "name": "5. SMOTE + threshold 0.3",
        "params": {
            "smote": True,
            "custom_threshold": 0.3,
        },
    },
    {
        "name": "6. SMOTE + threshold 0.2",
        "params": {
            "smote": True,
            "custom_threshold": 0.2,
        },
    },
    {
        "name": "7. SMOTE + threshold 0.1",
        "params": {
            "smote": True,
            "custom_threshold": 0.1,
        },
    },
    {
        "name": "8. SMOTE + threshold 0.09",
        "params": { 
            "smote": True,
            "custom_threshold": 0.09,
        },
    },
]


def run_single(name: str, X: pd.DataFrame, y: pd.Series, params: dict, results) -> dict:
    fbeta_scores, bac_scores = cross_validation_loop(X, y, **params)

    results[name] = {"fold_scores": bac_scores, "mean_bac": np.mean(bac_scores)}
    return results


def run_bac_experiment(X: pd.DataFrame, y: pd.Series) -> None:
    results = {}
    for exp in EXPERIMENTS:
        results = run_single(exp["name"], X, y, exp["params"], results)
    check_results(results)
    
    return results

def check_results(results):
    print("PODSUMOWANIE BAC dla wszystkich konfiguracji")
    for key, value in results.items():
        print(f"    {key}: {value["mean_bac"]:.3f}")