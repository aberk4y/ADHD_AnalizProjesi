import os
import joblib

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(UTILS_DIR, ".."))

MODEL_ROOT = os.path.join(BACKEND_DIR, "model")
REGISTRY_DIR = os.path.join(MODEL_ROOT, "registry")

SCENARIO_DIRS = {
    "A": "Senaryo_A_-_Tüm_Özellikler",
    "B": "Senaryo_B_-_Yasam_Aliskanliklari"
}

BEST_MODEL_PATH = os.path.join(MODEL_ROOT, "best_model.pkl")


def save_model(model, model_name, scenario_key):
    scenario_folder = SCENARIO_DIRS.get(scenario_key)
    if scenario_folder is None:
        raise ValueError("Geçersiz senaryo anahtarı (A veya B olmalı)")

    save_dir = os.path.join(REGISTRY_DIR, scenario_folder)
    os.makedirs(save_dir, exist_ok=True)

    file_name = model_name.replace(" ", "_") + ".pkl"
    save_path = os.path.join(save_dir, file_name)

    joblib.dump(model, save_path)
    return save_path


def load_models_by_scenario(scenario_key):
    models = {}

    scenario_folder = SCENARIO_DIRS.get(scenario_key)
    if scenario_folder is None:
        return models

    scenario_path = os.path.join(REGISTRY_DIR, scenario_folder)
    if not os.path.exists(scenario_path):
        return models

    for file in os.listdir(scenario_path):
        if file.endswith(".pkl"):
            model_name = file.replace(".pkl", "").replace("_", " ")
            model_path = os.path.join(scenario_path, file)
            models[model_name] = joblib.load(model_path)

    return models


def save_best_model(model):
    os.makedirs(MODEL_ROOT, exist_ok=True)
    joblib.dump(model, BEST_MODEL_PATH)


def load_best_model():
    if os.path.exists(BEST_MODEL_PATH):
        return joblib.load(BEST_MODEL_PATH)
    return None
