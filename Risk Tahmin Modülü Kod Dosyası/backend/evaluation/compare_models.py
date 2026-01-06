import sys
import os
import joblib
import pandas as pd

from backend.utils.model_registry import save_model, save_best_model
from backend.data_pipeline.preprocess import load_data, preprocess_full, preprocess_lifestyle
from backend.data_pipeline.split_data import split_dataset
from backend.models.logistic_model import get_model as get_log_model
from backend.models.random_forest_model import get_model as get_rf_model
from backend.models.svm_model import get_model as get_svm_model
from backend.models.gradient_boosting_model import get_model as get_gb_model
from backend.evaluation.metrics import evaluate_model
from backend.evaluation.plots import save_confusion_matrix
from backend.evaluation.bar_charts import save_metric_barchart
from backend.models.knn_model import get_model as get_knn_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
sys.path.append(BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "backend", "data", "adhd_data.csv")


RESULT_CACHE_PATH = os.path.join(BASE_DIR, "backend", "model", "model_results.pkl")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "backend", "model", "best_model.pkl")


def run_single_scenario(df, scenario_name, mode, scenario_key):

    if mode == "full":
        X, y = preprocess_full(df)
    elif mode == "lifestyle":
        X, y = preprocess_lifestyle(df)
    else:
        raise ValueError("Mode 'full' veya 'lifestyle' olmalidir!")

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    models = {
        "Logistic Regression": get_log_model(),
        "Random Forest": get_rf_model(),
        "SVM (RBF)": get_svm_model(),
        "Gradient Boosting": get_gb_model(),
        "KNN": get_knn_model(),
    }

    results = []
    trained_models = {}

    print(f"\n===== {scenario_name} — {mode.upper()} MODE =====")

    for name, model in models.items():
        print(f"Model egitiliyor: {name}")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        
        save_model(
           model=model,
           model_name=name,
           scenario_key=scenario_key
        ) 

        metrics = evaluate_model(y_test, preds)
        metrics["model"] = name
        metrics["scenario"] = scenario_name

        conf_path = save_confusion_matrix(
            metrics["confusion"],
            name,
            scenario_name
        )

        metrics["conf_matrix_path"] = conf_path

        results.append(metrics)
        trained_models[name] = model

    return pd.DataFrame(results), trained_models


def generate_all_barcharts(results_df):

    scenarios = results_df["scenario"].unique()
    for sc in scenarios:
        save_metric_barchart(results_df, sc)


def compare_models(data_path=DATA_PATH):

    df = load_data(data_path)

    full_results, full_models = run_single_scenario(df, "Senaryo_A_-_Tüm_Özellikler", mode="full", scenario_key="A")
    life_results, life_models = run_single_scenario(df, "Senaryo_B_-_Yasam_Aliskanliklari", mode="lifestyle", scenario_key="B")


    final_results = pd.concat([full_results, life_results], ignore_index=True)

    print("\nTüm Senaryolar:")
    print(final_results[["scenario", "model", "accuracy", "precision", "recall", "f1_score"]])

    
    best_row = final_results.sort_values(by="f1_score", ascending=False).iloc[0]
    scenario = best_row["scenario"]
    model_name = best_row["model"]


    best_model = (
        full_models[model_name] if "A" in scenario else life_models[model_name]
    )

    os.makedirs(os.path.join(BASE_DIR, "backend", "model"), exist_ok=True)

    joblib.dump(final_results, RESULT_CACHE_PATH)
    joblib.dump(best_model, BEST_MODEL_PATH)


    generate_all_barcharts(final_results)

    print("\n Sonuçlar ve en iyi model kaydedildi.")

    return final_results


if __name__ == "__main__":
    print("\n *** MODEL KARŞILAŞTIRMA BAŞLIYOR ***\n")
    results = compare_models()
    print("\n *** MODEL KARŞILAŞTIRMA TAMAMLANDI ***\n")
