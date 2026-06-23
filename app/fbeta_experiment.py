from cross_validation import cross_validation_loop_fbeta
import numpy as np

SMOTE_OPTIONS = {
    "smote": [False, True],
    "k_neighbors": [3, 5, 7],
    "sample_strategy": ["auto", 0.3, 0.5, 0.7]
}
WEIGHT_OPTIONS = [None, "balanced"]
RF_OPTIONS = {
    "rf_max_depth": [None, 5, 10, 20],
    "rf_min_samples_leaf": [1, 2, 5, 10],
    "rf_n_estimators": [50, 100, 200],
}
RSKF_OPTIONS = {
    "n_splits": [],
    "n_repeats": []
}
THRESHOLD_OPTIONS = [0.4, 0.5, 0.6]

def run_experiment_fbeta(X, y):
    results = {}
    results["basic_results"] = run_basics(X, y)
    check_results(results)
    return results

def run_basics(X, y):
    basic_results = {}
    for smote_option in SMOTE_OPTIONS["smote"]:
            if smote_option:
                for sample_option in SMOTE_OPTIONS["sample_strategy"]:
                    for k_neighbors_option in SMOTE_OPTIONS["k_neighbors"]:
                        fold_scores = cross_validation_loop_fbeta(X, y, smote=smote_option,  smote_sampling_strategy=sample_option, smote_k_neighbors=k_neighbors_option)
                        key = f"weight_None_smote_{smote_option}_sampling_{sample_option}_k_neighbors_{k_neighbors_option}"
                        basic_results[key] = {"fold_scores": fold_scores, "mean_fbeta": np.mean(fold_scores)}
            else:
                for weight_option in WEIGHT_OPTIONS:
                    fold_scores = cross_validation_loop_fbeta(X, y, smote=smote_option, class_weight=weight_option)
                    key = f"weight_{weight_option}_smote_{smote_option}"
                    basic_results[key] = {"fold_scores": fold_scores, "mean_fbeta": np.mean(fold_scores)}
    return basic_results

def run_weight_advanced(X, y):
    # weight_results = {}
    # for smote_option in SMOTE_OPTIONS:
    #     for weight_option in WEIGHT_OPTIONS:
    #         all_y_true, all_y_pred = cross_validation_loop(X, y, smote=smote_option, class_weight=weight_option)
    #         weight_results[f"weight_balanced"] = {"all_y_true": all_y_true, "all_y_pred": all_y_pred}
    # return weight_results
    pass

def check_results(results):
    if "basic_results" in results:
        basic_results = results["basic_results"]
        print("Basic results - checking influence of smote and weight with other parameters being default")
        for key, value in basic_results.items():
            print(f"    {key}: {value["mean_fbeta"]:.3f}")

    # for all_y_true, all_y_pred in results[0]:
    #     print("\n=== RAPORT Z REPEATED STRATIFIED K-FOLD + SMOTE ===")
    #     print(fbeta_score(all_y_true, all_y_pred, beta=1))
    #     # print(fbeta_score(all_y_true, all_y_pred, beta=2))
    #     # print(fbeta_score(all_y_true, all_y_pred, beta=0.5))