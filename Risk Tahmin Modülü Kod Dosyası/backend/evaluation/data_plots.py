import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "frontend", "static", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_class_distribution(df):
    path = os.path.join(PLOTS_DIR, "class_distribution.png")

    df["Diagnosis_Class"].value_counts().plot(
        kind="bar",
        color="steelblue"
    )
    plt.title("Sınıf Dağılımı")
    plt.xlabel("Sınıf")
    plt.ylabel("Örnek Sayısı")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return "plots/class_distribution.png"


def save_correlation_heatmap(df):
    path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")

    numeric_df = df.select_dtypes(include=["number"])

    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
    plt.title("Korelasyon Matrisi (Sayısal Değişkenler)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return "plots/correlation_heatmap.png"


def save_scatter_plot(df, x_col, y_col):
    file_name = f"scatter_{x_col}_vs_{y_col}.png"
    save_path = os.path.join(PLOTS_DIR, file_name)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue="Diagnosis_Class",
        palette="Set1",
        alpha=0.7
    )
    plt.title(f"{x_col} vs {y_col}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return f"plots/{file_name}"
