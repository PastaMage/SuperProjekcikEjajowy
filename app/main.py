from pathlib import Path
import pandas as pd
from plot_generation import plot_depression_balance, plot_numeric_distributions, plot_platform_usage


DATA_PATH = Path("./Teen_Mental_Health_Dataset.csv")

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Data shape: {df.shape}")
    print(f"Max null count per column: {df.isnull().sum().max()}")
    return df

def main() -> None:
    df = load_data(DATA_PATH)
    plot_platform_usage(df)
    plot_numeric_distributions(df)
    plot_depression_balance(df)


if __name__ == "__main__":
    main()