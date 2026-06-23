from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

NUM_COLS = [
    'age', 'daily_social_media_hours', 'sleep_hours', 'screen_time_before_sleep',
    'academic_performance', 'physical_activity', 'stress_level', 'anxiety_level', 'addiction_level'
]
ORDINAL_COLS = ['social_interaction_level']
NOMINAL_COLS = ['gender', 'platform_usage']

def prep_data_pipeline(df: pd.DataFrame):

    social_map = {'low': 0, 'medium': 1, 'high': 2}
    for col in ORDINAL_COLS:
        df[col] = df[col].map(social_map)

    for col in NOMINAL_COLS:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

    X = df.drop('depression_label', axis=1)
    y = df['depression_label']

    scaler = StandardScaler()
    
    X[NUM_COLS] = scaler.fit_transform(X[NUM_COLS])
    
    print(f"Dane przygotowane ręcznie. Kształt X: {X.shape}")
    print(f"Przykładowe dane:\n{X.head(3)}")
    
    return X, y