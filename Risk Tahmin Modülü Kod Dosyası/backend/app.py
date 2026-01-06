import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from backend.evaluation.compare_models import compare_models
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from backend.utils.model_registry import load_models_by_scenario
from backend.data_pipeline.preprocess import (
    preprocess_input_full,
    preprocess_input_lifestyle,
    load_data
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))


DATA_PATH = os.path.join(APP_DIR, "data", "adhd_data.csv")
model_cache = None

app = Flask(
    __name__,
    template_folder=os.path.join(APP_DIR, "../frontend/templates"),
    static_folder=os.path.join(APP_DIR, "../frontend/static")
)


def clamp(val, low=0, high=100):
    try:
        return max(low, min(high, int(round(float(val)))))
    except Exception:
        return max(low, min(high, int(val)))


def clamp_by_class(risk_value, predicted_class):
   
    limits = {
        0: (0, 30),
        1: (30, 55),
        2: (55, 80),
        3: (80, 100)
    }
    low, high = limits.get(int(predicted_class), (0, 100))
    return clamp(risk_value, low, high)


def risk_score_scenario_a(predicted_class, q1_avg, q2_avg):
    
    base_map = {0: 10, 1: 35, 2: 60, 3: 85}
    base = base_map.get(int(predicted_class), 10)

    q1 = float(q1_avg) if q1_avg is not None else 0.0
    q2 = float(q2_avg) if q2_avg is not None else 0.0

    question_risk = ((q1 + q2) / 6.0) * 25.0

    raw_risk = base + question_risk
    return clamp_by_class(raw_risk, predicted_class)


def risk_score_scenario_b(predicted_class, summary):
    
    base_map = {0: 20, 1: 40, 2: 60, 3: 80}
    base = base_map.get(int(predicted_class), 20)

    sleep = float(summary.get("Sleep_Hours", 0))
    phone = float(summary.get("Daily_Phone_Usage_Hours", 0))
    anxiety = float(summary.get("Anxiety_Depression_Levels", 0))
    learning = float(summary.get("Learning_Difficulties", 0))
    org = float(summary.get("Difficulty_Organizing_Tasks", 0))
    focus = float(summary.get("Focus_Score_Video", 0))

    activity = float(summary.get("Daily_Activity_Hours", 0))
    walking = float(summary.get("Daily_Walking_Running_Hours", 0))
    coffee = float(summary.get("Daily_Coffee_Tea_Consumption", 0))
    family = float(summary.get("Family_History", 0))

    score = 0.0


    score += max(0.0, (7.0 - sleep)) * 4.0
    score += phone * 2.0
    score += anxiety * 5.0
    score += learning * 6.0
    score += org * 4.0
    score += focus * 4.0
    score -= activity * 2.0
    score -= walking * 2.0
    score += max(0.0, coffee - 1.0) * 2.0
    score += family * 6.0

    raw_risk = base + score
    return clamp_by_class(raw_risk, predicted_class)


def process_form_data(form):
    try:

        q1_scores = [int(form.get(f"Q1_{i}", 0) or 0) for i in range(1, 10)]
        q2_scores = [int(form.get(f"Q2_{i}", 0) or 0) for i in range(1, 10)]

        q1_avg = float(np.mean(q1_scores)) if q1_scores else None
        q2_avg = float(np.mean(q2_scores)) if q2_scores else None

        data = {
            "Age": [int(form["age"])],
            "Sleep_Hours": [float(form["sleep_hours"])],
            "Daily_Phone_Usage_Hours": [float(form["phone_hours"])],

            "Daily_Activity_Hours": [float(form["daily_activity"])],
            "Daily_Walking_Running_Hours": [float(form["walking_hours"])],
            "Daily_Coffee_Tea_Consumption": [float(form["coffee_tea"])],

            "Learning_Difficulties": [int(form["learning_diff"])],
            "Anxiety_Depression_Levels": [int(form["anxiety"])],
            "Educational_Level": [int(form["education"])],
            "Family_History": [int(form["family_history"])],

            "Difficulty_Organizing_Tasks": [int(form["org"])],
        }

        for i in range(1, 10):
            data[f"Q1_{i}"] = [q1_scores[i - 1]]
            data[f"Q2_{i}"] = [q2_scores[i - 1]]


        data["Focus_Score_Video"] = [q1_avg if q1_avg is not None else 0.0]

        df = pd.DataFrame(data)

        return df, q1_avg, q2_avg

    except Exception as e:
        print("Veri işleme hatası:", e)
        return None, None, None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    predictions = {}
    probabilities = {}

    df_raw, q1_avg, q2_avg = process_form_data(request.form)
    if df_raw is None:
        return render_template("result.html", error="Girdi hatası!")


    has_q1 = any(int(request.form.get(f"Q1_{i}", 0) or 0) > 0 for i in range(1, 10))
    has_q2 = any(int(request.form.get(f"Q2_{i}", 0) or 0) > 0 for i in range(1, 10))pyt
    scenario = "A" if (has_q1 and has_q2) else "B"

    models = load_models_by_scenario(scenario)

    print("Seçilen senaryo:", scenario)
    print("Yüklenen modeller:", list(models.keys()))

    if not models:
        return render_template("result.html", error="Bu senaryo için model yok.")


    if scenario == "A":
        ref_model = next(iter(models.values()))
        ref_cols = list(ref_model.feature_names_in_)
        df_input = preprocess_input_full(df_raw, ref_cols)
    else:
        df_input = preprocess_input_lifestyle(df_raw)


    for name, model in models.items():
        try:
            pred = model.predict(df_input)[0]
            proba = model.predict_proba(df_input)[0]
            predictions[name] = int(pred)
            probabilities[name] = proba.tolist()
        except Exception as e:
            print(f"{name} hata:", e)

    if not predictions:
        return render_template("result.html", error="Tahmin yapılamadı.")


    best_model = max(
        predictions,
        key=lambda m: probabilities[m][predictions[m]]
    )

    predicted_class = predictions[best_model]
    confidence = float(probabilities[best_model][predicted_class])
    summary = df_raw.to_dict(orient="records")[0]

    
    if scenario == "A":
        risk_value = risk_score_scenario_a(predicted_class, q1_avg, q2_avg)
    else:
        risk_value = risk_score_scenario_b(predicted_class, summary)

    return render_template(
        "result.html",
        predictions=predictions,
        probabilities=probabilities,
        best_model=best_model,
        prediction=predicted_class,
        confidence=confidence,
        risk_value=risk_value,
        q1_avg=q1_avg,
        q2_avg=q2_avg,
        scenario=scenario,
        summary=summary
    )


@app.route("/models")
def models_page():
    global model_cache

    if model_cache is None:
        print(">>> Modeller ilk kez eğitiliyor...")
        results_df = compare_models(DATA_PATH)  
        model_cache = results_df
    else:
        print(">>> Cache'den model sonuçları alındı.")
        results_df = model_cache

    scenario_a_df = results_df[results_df["scenario"].str.contains("Senaryo_A")]
    scenario_b_df = results_df[results_df["scenario"].str.contains("Senaryo_B")]

    return render_template(
        "models.html",
        scenario_a=scenario_a_df.to_dict(orient="records"),
        scenario_b=scenario_b_df.to_dict(orient="records")
    )


@app.route("/insights", methods=["GET", "POST"])
def insights_page():
    from backend.evaluation.data_plots import (
        plot_class_distribution,
        save_correlation_heatmap,
        save_scatter_plot
    )

    df = load_data(DATA_PATH)  

    strategy = request.args.get("missing_strategy", "mean")
    if strategy == "mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif strategy == "median":
        df = df.fillna(df.median(numeric_only=True))
    elif strategy == "drop":
        df = df.dropna()

    scaling = request.args.get("scaling", "none")
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if scaling in ["standard", "minmax"] and len(numeric_columns) > 0:
        cols_to_scale = [c for c in numeric_columns if c != "Diagnosis_Class"]

        if scaling == "standard":
            scaler = StandardScaler()
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        elif scaling == "minmax":
            scaler = MinMaxScaler()
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    scatter_plot = None
    selected_x = None
    selected_y = None

    if request.method == "POST":
        selected_x = request.form.get("x_col")
        selected_y = request.form.get("y_col")
        if selected_x and selected_y:
            scatter_plot = save_scatter_plot(df, selected_x, selected_y)

    class_plot = plot_class_distribution(df)
    heatmap_plot = save_correlation_heatmap(df)

    return render_template(
        "insights.html",
        columns=df.columns.tolist(),
        numeric_columns=numeric_columns,
        missing=df.isnull().sum().to_dict(),
        class_dist=df["Diagnosis_Class"].value_counts().to_dict(),
        class_plot=class_plot,
        heatmap_plot=heatmap_plot,
        scatter_plot=scatter_plot,
        selected_x=selected_x,
        selected_y=selected_y,
        selected_strategy=strategy,
        selected_scaling=scaling
    )


@app.route("/analysis", methods=["GET"])
def analysis_page():
    return render_template("analysis.html")


if __name__ == "__main__":
    app.run(debug=True)
