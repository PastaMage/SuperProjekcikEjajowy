import pandas as pd
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
]


def run_single(name: str, X: pd.DataFrame, y: pd.Series, params: dict) -> dict:
    print(f"Eksperyment BAC: {name}")
    print(f"Parametry:   {params if params else 'domyślne'}")

    y_pred, y_true = cross_validation_loop(X, y, **params)

    bac = balanced_accuracy_score(y_true, y_pred)

    print(f"\nBAC (Balanced Accuracy): {bac:.4f}")
    print("\nRaport klasyfikacji:")
    print(classification_report(y_true, y_pred, digits=4))
    print("Macierz pomyłek:")
    print(confusion_matrix(y_true, y_pred))
    print()

    return {"name": name, "bac": bac, "params": params}


def run_bac_experiment(X: pd.DataFrame, y: pd.Series) -> None:
    results = []
    for exp in EXPERIMENTS:
        result = run_single(exp["name"], X, y, exp["params"])
        results.append(result)
    print("PODSUMOWANIE BAC dla wszystkich konfiguracji")
    for idx, r in enumerate(results, start=1):
        print(f"  {idx}. {r['name']:<35} BAC = {r['bac']:.4f}")

    return results