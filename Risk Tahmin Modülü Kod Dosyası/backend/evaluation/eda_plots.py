import os
import seaborn as sns
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
FRONTEND_STATIC_DIR = os.path.join(PROJECT_ROOT, "frontend", "static")

def save_correlation_heatmap(df):
    save_dir = os.path.join(FRONTEND_STATIC_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    sns.heatmap(corr, cmap="coolwarm", linewidths=0.5)
    plt.title("Korelasyon Matrisi")

    path = os.path.join(save_dir, "correlation_heatmap.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Korelasyon ısı haritası kaydedildi: {path}")

    return "plots/correlation_heatmap.png"
