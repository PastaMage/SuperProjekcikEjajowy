from scipy.stats import wilcoxon
import numpy as np

ALPHA = 0.05

def analyze_wilcoxon_fbeta(data):
    print(f"\n{'='*105}")
    print(f" TURNIEJ STATYSTYCZNY (Każdy z każdym) - F-beta")
    print(f"{'='*105}")
    
    conf_names = list(data.keys())
    n_models = len(conf_names)

    tournament = {name: {"wins": 0, "losses": 0, "ties": 0, "mean": data[name]["mean_fbeta"]} for name in conf_names}

    for i in range(n_models):
        for j in range(i + 1, n_models):
            c1 = conf_names[i]
            c2 = conf_names[j]

            scores1 = data[c1]["fold_scores"]
            scores2 = data[c2]["fold_scores"]

            try:
                stat, p_value = wilcoxon(scores1, scores2)
            except ValueError:
                p_value = 1.0 
            if p_value < ALPHA:
                if tournament[c1]["mean"] > tournament[c2]["mean"]:
                    tournament[c1]["wins"] += 1
                    tournament[c2]["losses"] += 1
                else:
                    tournament[c2]["wins"] += 1
                    tournament[c1]["losses"] += 1
            else:
                tournament[c1]["ties"] += 1
                tournament[c2]["ties"] += 1

    ranked_models = sorted(tournament.items(), key=lambda x: (x[1]["wins"], x[1]["mean"]), reverse=True)

    print(f"{'Pozycja':<7} | {'Konfiguracja':<55} | {'Śr. F-beta':<10} | {'Wygrane':<7} | {'Remisy':<6} | {'Przegrane':<9} ")
    print("-" * 105)

    for rank, (name, stats) in enumerate(ranked_models, 1):
        print(f"#{rank:<6} | {name:<55} | {stats['mean']:<10.4f} | {stats['wins']:<7} | {stats['ties']:<6} | {stats['losses']:<9}")
        
    print(f"{'='*105}")

def analyze_wilcoxon_bac(data):
    print(f"\n{'='*105}")
    print(f" TURNIEJ STATYSTYCZNY (Każdy z każdym) - F-beta")
    print(f"{'='*105}")
    
    conf_names = list(data.keys())
    n_models = len(conf_names)

    tournament = {name: {"wins": 0, "losses": 0, "ties": 0, "mean": data[name]["mean_bac"]} for name in conf_names}

    for i in range(n_models):
        for j in range(i + 1, n_models):
            c1 = conf_names[i]
            c2 = conf_names[j]

            scores1 = data[c1]["fold_scores"]
            scores2 = data[c2]["fold_scores"]

            try:
                stat, p_value = wilcoxon(scores1, scores2)
            except ValueError:
                p_value = 1.0 
            if p_value < ALPHA:
                if tournament[c1]["mean"] > tournament[c2]["mean"]:
                    tournament[c1]["wins"] += 1
                    tournament[c2]["losses"] += 1
                else:
                    tournament[c2]["wins"] += 1
                    tournament[c1]["losses"] += 1
            else:
                tournament[c1]["ties"] += 1
                tournament[c2]["ties"] += 1

    ranked_models = sorted(tournament.items(), key=lambda x: (x[1]["wins"], x[1]["mean"]), reverse=True)

    print(f"{'Pozycja':<7} | {'Konfiguracja':<55} | {'Śr. bac':<10} | {'Wygrane':<7} | {'Remisy':<6} | {'Przegrane':<9} ")
    print("-" * 105)

    for rank, (name, stats) in enumerate(ranked_models, 1):
        print(f"#{rank:<6} | {name:<55} | {stats['mean']:<10.4f} | {stats['wins']:<7} | {stats['ties']:<6} | {stats['losses']:<9}")
        
    print(f"{'='*105}")