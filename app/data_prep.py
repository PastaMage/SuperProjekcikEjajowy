from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

NUM_COLS = [
    'age', 'daily_social_media_hours', 'sleep_hours', 'screen_time_before_sleep',
    'academic_performance', 'physical_activity', 'stress_level', 'anxiety_level', 'addiction_level'
]

def prep_data(df: pd.DataFrame):
    df = change_data_to_numeric(df)
    X = df.drop('depression_label', axis=1)
    y = df['depression_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = scale_numerics(X_train, X_test)
    return X_train, X_test, y_train, y_test

def scale_numerics(X_train, X_test):
    scaler = StandardScaler()

    X_train[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
    X_test[NUM_COLS] = scaler.transform(X_test[NUM_COLS])
    print(f"Scale Numerics X_train:\n{X_train[0:5]}\nScale Numerics X_test:\n{X_test[0:5]}")

    return {"X_train": X_train, "X_test": X_test}

def change_data_to_numeric(df: pd.DataFrame):
    social_map = {'low': 0, 'medium': 1, 'high': 2}
    df['social_interaction_level'] = df['social_interaction_level'].map(social_map)

    df = pd.get_dummies(df, columns=['gender', 'platform_usage'], drop_first=True)
    print(f"Data to Numeric:\n{df[0:5]}")
    return df