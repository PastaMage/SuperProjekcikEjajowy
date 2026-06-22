from pathlib import Path
import pandas as pd
from plot_generation import plot_depression_balance, plot_numeric_distributions, plot_platform_usage
from data_prep import scale_numerics, change_data_to_numeric, prep_data
from sklearn.model_selection import train_test_split

DATA_PATH = Path("./Teen_Mental_Health_Dataset.csv")

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Data shape: {df.shape}")
    print(f"Max null count per column: {df.isnull().sum().max()}")
    return df



def main() -> None:
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = prep_data(df)
    

if __name__ == "__main__":
    main()