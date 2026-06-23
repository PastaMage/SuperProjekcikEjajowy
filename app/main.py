from pathlib import Path
import pandas as pd
from data_prep import prep_data_pipeline
from cross_validation import cross_validation_loop


DATA_PATH = Path("./Teen_Mental_Health_Dataset.csv")

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Data shape: {df.shape}")
    print(f"Max null count per column: {df.isnull().sum().max()}")
    return df

def main() -> None:
    df = load_data(DATA_PATH)
    X, y = prep_data_pipeline(df)
    cross_validation_loop(X, y)


if __name__ == "__main__":
    main()  