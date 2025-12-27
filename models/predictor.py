import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from joblib import load
from config import MODEL_PATH


# ==================================================
# Mapping kategori ke numerik (HARUS SAMA DENGAN TRAINING)
# ==================================================
MAP = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}


# ==================================================
# FITUR HARUS PERSIS SAMA DENGAN train_model.py
# ==================================================
FEATURES = [
    "modal_min",
    "modal_max",
    "persaingan",
    "complexity",
    "potensi_margin",
    "kebutuhan_keahlian",
    "daya_beli_target"
]


# ==================================================
# Load model
# ==================================================
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        "Model belum ada. Jalankan dulu: python models/train_model.py"
    )

model = load(MODEL_PATH)


# ==================================================
# Predict probabilitas (1 baris)
# ==================================================
def predict_probability(row: dict):
    """
    row: satu baris dataset (dict)
    Dipakai kalau mau predict per-usaha
    """

    data = {
        "modal_min": row["modal_min"],
        "modal_max": row["modal_max"],
        "persaingan": MAP[row["persaingan"]],
        "complexity": MAP[row["complexity"]],
        "potensi_margin": MAP[row["potensi_margin"]],
        "kebutuhan_keahlian": MAP[row["kebutuhan_keahlian"]],
        "daya_beli_target": MAP[row["daya_beli_target"]],
    }

    X = pd.DataFrame([data], columns=FEATURES)

    prob = model.predict_proba(X)[0][1]
    return round(prob * 100, 2)
