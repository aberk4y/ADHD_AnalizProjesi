import os
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
FRONTEND_STATIC_DIR = os.path.join(PROJECT_ROOT, "frontend", "static")

def save_confusion_matrix(conf_matrix, model_name, scenario_name):
    save_dir = os.path.join(FRONTEND_STATIC_DIR, "confusion_matrices")
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"{scenario_name}_{model_name}_cm.png".replace(" ", "_")
    save_path = os.path.join(save_dir, file_name)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
    plt.title(f"{scenario_name} - {model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

    return f"confusion_matrices/{file_name}"
