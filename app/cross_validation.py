from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

NUM_COLS = [
    'age', 'daily_social_media_hours', 'sleep_hours', 'screen_time_before_sleep',
    'academic_performance', 'physical_activity', 'stress_level', 'anxiety_level', 'addiction_level'
]

def cross_validation_loop(X, y):
    forest_clf = RandomForestClassifier(random_state=42, class_weight='balanced')

    smote = SMOTE(random_state=42)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', smote),
        ('classifier', forest_clf)
    ])


    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    all_y_true = []
    all_y_pred = []

    for train_index, val_index in rskf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        
        pipeline.fit(X_train_fold, y_train_fold)
        
        y_pred_fold = pipeline.predict(X_val_fold)

        all_y_true.extend(y_val_fold)
        all_y_pred.extend(y_pred_fold)

    return all_y_pred, all_y_true

    