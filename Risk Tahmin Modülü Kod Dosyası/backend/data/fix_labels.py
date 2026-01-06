import pandas as pd


DATA_PATH = "backend/data/adhd_data.csv"
df = pd.read_csv(DATA_PATH)

print("Düzeltme ÖNCESİ sınıf dağılımı:")
print(df["Diagnosis_Class"].value_counts())

df["Diagnosis_Class"] = df["Diagnosis_Class"].replace({
    1: 2,
    2: 1
})

print("\nDüzeltme SONRASI sınıf dağılımı:")
print(df["Diagnosis_Class"].value_counts())

df.to_csv(DATA_PATH, index=False)

print("\n Diagnosis_Class etiketleri düzeltildi.")
