import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "adhd_data.csv")
PLOT_DIR = os.path.join(BASE_DIR, "..", "frontend", "static", "plots")

os.makedirs(PLOT_DIR, exist_ok=True)


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    return df


def get_column_info(df):
    return df.dtypes.to_dict()


def get_missing_values(df):
    return df.isnull().sum().to_dict()


def plot_class_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df["Diagnosis_Class"], palette="viridis")
    plt.title("Sınıf Dağılımı")
    path = os.path.join(PLOT_DIR, "class_distribution.png")
    plt.savefig(path)
    plt.close()
    return "class_distribution.png"


def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['number'])
    sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
    plt.title("Korelasyon Matrisi")
    path = os.path.join(PLOT_DIR, "correlation_heatmap.png")
    plt.savefig(path)
    plt.close()
    return "correlation_heatmap.png"


def plot_histograms(df):
    numeric_cols = ["Age", "Sleep_Hours", "Daily_Phone_Usage_Hours"]

    filenames = []

    for col in numeric_cols:
        if col in df.columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], kde=True, color="blue")
            plt.title(f"{col} Histogram")
            filename = f"hist_{col}.png"
            plt.savefig(os.path.join(PLOT_DIR, filename))
            plt.close()
            filenames.append(filename)

    return filenames
