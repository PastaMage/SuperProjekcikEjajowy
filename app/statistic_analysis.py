from scipy.stats import wilcoxon
import numpy as np

ALPHA = 0.05

def analyze_wilcoxon(data):
    print(f"\n{'='*105}")
    print(f" TURNIEJ STATYSTYCZNY (Każdy z każdym) - F-beta")
    print(f"{'='*105}")
    
    model_names = list(data.keys())
    n_models = len(model_names)

    tournament = {name: {"wins": 0, "losses": 0, "ties": 0, "mean": data[name]["mean_fbeta"]} for name in model_names}

    for i in range(n_models):
        for j in range(i + 1, n_models):
            m1 = model_names[i]
            m2 = model_names[j]

            scores1 = data[m1]["fold_scores"]
            scores2 = data[m2]["fold_scores"]

            try:
                stat, p_value = wilcoxon(scores1, scores2)
            except ValueError:
                p_value = 1.0 
            if p_value < 0.05:
                if tournament[m1]["mean"] > tournament[m2]["mean"]:
                    tournament[m1]["wins"] += 1
                    tournament[m2]["losses"] += 1
                else:
                    tournament[m2]["wins"] += 1
                    tournament[m1]["losses"] += 1
            else:
                tournament[m1]["ties"] += 1
                tournament[m2]["ties"] += 1

    ranked_models = sorted(tournament.items(), key=lambda x: (x[1]["wins"], x[1]["mean"]), reverse=True)

    print(f"{'Pozycja':<7} | {'Nazwa Modelu / Konfiguracja':<55} | {'Śr. F-beta':<10} | {'Wygrane':<7} | {'Remisy':<6} | {'Przegrane':<9}")
    print("-" * 105)

    for rank, (name, stats) in enumerate(ranked_models, 1):
        print(f"#{rank:<6} | {name:<55} | {stats['mean']:<10.4f} | {stats['wins']:<7} | {stats['ties']:<6} | {stats['losses']:<9}")
        
    print(f"{'='*105}")
    # print("WNIOSKI:")
    
    # best_name = ranked_models[0][0]
    # best_stats = ranked_models[0][1]
    # print(f"🥇 Najlepszy model to: {best_name}")
    # print(f"   Wygrał {best_stats['wins']} statystycznych pojedynków z innymi modelami.")

    # # Znalezienie modeli, które w bezpośrednim pojedynku statystycznie zremisowały z liderem
    # tied_with_leader = []
    # scores_leader = results_dict[best_name]["fold_scores"]
    
    # for name, stats in ranked_models[1:]:
    #     scores_other = results_dict[name]["fold_scores"]
    #     try:
    #         _, p = wilcoxon(scores_leader, scores_other)
    #     except ValueError:
    #         p = 1.0
            
    #     if p >= 0.05:
    #         tied_with_leader.append(name)

    # if tied_with_leader:
    #     print("\n🤝 Uwaga! Następujące konfiguracje statystycznie REMISUJĄ z liderem (brak różnicy, p >= 0.05):")
    #     for tied in tied_with_leader:
    #         print(f"   - {tied} (Śr. F-beta: {results_dict[tied]['mean_fbeta']:.4f})")
    #     print("Najlepiej wybrać z tej listy konfigurację najprostszą obliczeniowo (np. z domyślnym 'auto' lub mniejszą ilością sąsiadów 'k_neighbors').")
    # else:
    #     print("\n👑 Lider jest absolutnym zwycięzcą – statystycznie pokonał wszystkie inne warianty w walce jeden na jednego!")
