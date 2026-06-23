from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from typing import Any
from sklearn.metrics import fbeta_score

NUM_COLS = [
    'age', 'daily_social_media_hours', 'sleep_hours', 'screen_time_before_sleep',
    'academic_performance', 'physical_activity', 'stress_level', 'anxiety_level', 'addiction_level'
]
BETA_VALUE = 0.5

def cross_validation_loop(
    X,
    y, 
    class_weight=None, 
    smote=False, 
    smote_k_neighbors=5,
    smote_sampling_strategy="auto",
    n_splits=5, 
    n_repeats=3, 
    rf_max_depth=None, 
    rf_min_samples_leaf=1,
    rf_n_estimators=100,
    custom_threshold=0.5):

    forest_clf = RandomForestClassifier(
        random_state=42, 
        class_weight=class_weight,
        max_depth=rf_max_depth,
        min_samples_leaf=rf_min_samples_leaf,
        n_estimators=rf_n_estimators
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS)
        ],
        remainder='passthrough'
    )

    steps: list[tuple[str, Any]] = [('preprocessor', preprocessor)]
    
    if smote:
        steps.append(('smote', SMOTE(random_state=42, k_neighbors=smote_k_neighbors, sampling_strategy=smote_sampling_strategy)))
        
    steps.append(('classifier', forest_clf))
    pipeline = Pipeline(steps)

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    all_y_true = []
    all_y_pred = []

    for train_index, val_index in rskf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        
        pipeline.fit(X_train_fold, y_train_fold)
        
        probabilities = pipeline.predict_proba(X_val_fold)[:, 1]
        y_pred_fold = (probabilities >= custom_threshold).astype(int)

        all_y_true.extend(y_val_fold)
        all_y_pred.extend(y_pred_fold)

    return all_y_pred, all_y_true

def cross_validation_loop_fbeta(
    X,
    y, 
    class_weight=None, 
    smote=False, 
    smote_k_neighbors=5,
    smote_sampling_strategy="auto",
    n_splits=5, 
    n_repeats=3, 
    rf_max_depth=None, 
    rf_min_samples_leaf=1,
    rf_n_estimators=100,
    custom_threshold=0.5):

    forest_clf = RandomForestClassifier(
        random_state=42, 
        class_weight=class_weight,
        max_depth=rf_max_depth,
        min_samples_leaf=rf_min_samples_leaf,
        n_estimators=rf_n_estimators
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS)
        ],
        remainder='passthrough'
    )

    steps: list[tuple[str, Any]] = [('preprocessor', preprocessor)]
    
    if smote:
        steps.append(('smote', SMOTE(random_state=42, k_neighbors=smote_k_neighbors, sampling_strategy=smote_sampling_strategy)))
        
    steps.append(('classifier', forest_clf))
    pipeline = Pipeline(steps)

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    fold_fbeta_scores = []

    for train_index, val_index in rskf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        
        pipeline.fit(X_train_fold, y_train_fold)
        
        probabilities = pipeline.predict_proba(X_val_fold)[:, 1]
        y_pred_fold = (probabilities >= custom_threshold).astype(int)

        fold_score = fbeta_score(y_val_fold, y_pred_fold, beta=BETA_VALUE)
        fold_fbeta_scores.append(fold_score)

    return fold_fbeta_scores