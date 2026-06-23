from pathlib import Path
import pandas as pd
from plot_generation import plot_depression_balance, plot_numeric_distributions, plot_platform_usage
from data_prep import prep_data_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = Path("./Teen_Mental_Health_Dataset.csv")

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Data shape: {df.shape}")
    print(f"Max null count per column: {df.isnull().sum().max()}")
    return df



def main() -> None:
    df = load_data(DATA_PATH)
    X, y = prep_data_pipeline(df)

    forest_clf = RandomForestClassifier(class_weight='balanced', random_state=42)

    # forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=5, scoring='accuracy')

    # print(f"Regresja Logistyczna - średnia trafność CV: {np.mean(log_scores):.4f}")
    # print(f"Las Losowy - średnia trafność CV: {np.mean(forest_scores):.4f}")

    # y_train_pred_log = cross_val_predict(log_reg, X_train, y_train, cv=5)
    # y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train, cv=5)

    # print("=== REGRESJA LOGISTYCZNA ===")
    # print("Macierz pomyłek:")
    # print(confusion_matrix(y_train, y_train_pred_log))
    # print("\nRaport klasyfikacji:")
    # print(classification_report(y_train, y_train_pred_log))

    # print("\n=== LAS LOSOWY ===")
    # print("Macierz pomyłek:")
    # print(confusion_matrix(y_train, y_train_pred_forest))
    # print("\nRaport klasyfikacji:")
    # print(classification_report(y_train, y_train_pred_forest))


if __name__ == "__main__":
    main()  