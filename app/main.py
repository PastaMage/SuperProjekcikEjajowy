from pathlib import Path
import pandas as pd
from data_prep import prep_data_pipeline
from bac_experiment import run_bac_experiment
from fbeta_experiment import run_experiment_fbeta
from statistic_analysis import analyze_wilcoxon_fbeta, analyze_wilcoxon_bac

DATA_PATH = Path("./Teen_Mental_Health_Dataset.csv")

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Data shape: {df.shape}")
    print(f"Max null count per column: {df.isnull().sum().max()}")
    return df

def main() -> None:
    df = load_data(DATA_PATH)
    X, y = prep_data_pipeline(df)
    f_beta_results = run_experiment_fbeta(X, y)
    analyze_wilcoxon_fbeta(f_beta_results["basic_results"])
    bac_results = run_bac_experiment(X, y)
    analyze_wilcoxon_bac(bac_results)


if __name__ == "__main__":
    main()  