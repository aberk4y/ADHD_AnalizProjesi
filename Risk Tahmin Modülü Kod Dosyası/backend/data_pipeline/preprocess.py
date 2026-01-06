import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("Veri başarıyla yüklendi.")
    return df


def encode_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    categorical_columns = ["Educational_Level", "Family_History", "Gender"]

    for col in categorical_columns:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df.fillna(0, inplace=True)
    return df


def preprocess_full(df: pd.DataFrame):
    df = encode_and_clean(df)

    target_column = "Diagnosis_Class"
    X = df.drop(columns=[target_column])
    y = df[target_column]

    print("Ön işleme tamamlandı (FULL MODE)")
    return X, y


def preprocess_lifestyle(df: pd.DataFrame):
    df = encode_and_clean(df)

    target_column = "Diagnosis_Class"

    lifestyle_cols = [
        "Age",
        "Sleep_Hours",
        "Daily_Phone_Usage_Hours",
        "Daily_Activity_Hours",
        "Daily_Walking_Running_Hours",
        "Daily_Coffee_Tea_Consumption",
        "Learning_Difficulties",
        "Anxiety_Depression_Levels",
        "Educational_Level",
        "Family_History",
        "Difficulty_Organizing_Tasks",
        "Focus_Score_Video"
    ]

    lifestyle_cols = [c for c in lifestyle_cols if c in df.columns]

    X = df[lifestyle_cols]
    y = df[target_column]

    print("Ön işleme tamamlandı (LIFESTYLE MODE)")
    return X, y


def preprocess_input_full(input_df: pd.DataFrame, reference_columns: list):
    
    df = input_df.copy()
    df = encode_and_clean(df)

    for col in reference_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[reference_columns]
    return df


def preprocess_input_lifestyle(input_df: pd.DataFrame):
   
    lifestyle_cols = [
        "Age",
        "Sleep_Hours",
        "Daily_Phone_Usage_Hours",
        "Daily_Activity_Hours",
        "Daily_Walking_Running_Hours",
        "Daily_Coffee_Tea_Consumption",
        "Learning_Difficulties",
        "Anxiety_Depression_Levels",
        "Educational_Level",
        "Family_History",
        "Difficulty_Organizing_Tasks",
        "Focus_Score_Video"
    ]
    
    
    df = input_df.copy()
    df = encode_and_clean(df)

    lifestyle_cols = [c for c in lifestyle_cols if c in df.columns]
    df = df[lifestyle_cols]
    return df
