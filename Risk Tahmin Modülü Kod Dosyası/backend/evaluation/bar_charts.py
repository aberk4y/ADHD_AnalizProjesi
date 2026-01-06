import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
FRONTEND_STATIC_DIR = os.path.join(PROJECT_ROOT, "frontend", "static")
MODEL_CHARTS_DIR = os.path.join(FRONTEND_STATIC_DIR, "model_charts")
os.makedirs(MODEL_CHARTS_DIR, exist_ok=True)


def _safe_filename(text: str) -> str:
    
    text = text.strip().replace(" ", "_")
    text = re.sub(r"[^a-zA-Z0-9_\-]", "", text)
    return text


def save_metric_barchart(df, scenario_name):
    
    df_s = df[df["scenario"] == scenario_name].copy()

    metrics = ["accuracy", "precision", "recall", "f1_score"]
    saved_paths = {}

    safe_scenario = _safe_filename(scenario_name)

    for metric in metrics:
        if metric not in df_s.columns:
            continue

        plt.figure(figsize=(7, 4))

        df_sorted = df_s.sort_values(by=metric, ascending=False)

        plt.bar(df_sorted["model"], df_sorted[metric])
        plt.title(f"{scenario_name} – {metric.upper()}")
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.xticks(rotation=15)

        plt.tight_layout()

        file_name = f"{safe_scenario}_{metric}.png"
        save_path = os.path.join(MODEL_CHARTS_DIR, file_name)

        plt.savefig(save_path)
        plt.close()

        saved_paths[metric] = f"model_charts/{file_name}"

    return saved_paths
