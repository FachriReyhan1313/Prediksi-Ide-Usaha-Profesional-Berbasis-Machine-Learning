import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import pandas as pd
from datetime import datetime
from joblib import dump

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.calibration import CalibratedClassifierCV

from config import DATASET_PATH, MODEL_PATH


# ==================================================
# Header
# ==================================================
print("====================================================")
print(" TRAINING MODEL PREDIKSI IDE USAHA (CALIBRATED) ")
print("====================================================")


# ==================================================
# 1. Load Dataset
# ==================================================
df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded            : {len(df)} records")


# ==================================================
# 2. 🔥 GENERATE PSEUDO-LABEL (INI YANG KURANG)
# ==================================================
df["status_keberhasilan"] = (
    (df["potensi_margin"].isin(["Medium", "High"])) &
    (df["daya_beli_target"].isin(["Medium", "High"])) &
    (df["complexity"] != "High")
).astype(int)


# ==================================================
# 3. Preprocessing & Encoding
# ==================================================
MAP = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

for col in [
    "persaingan",
    "complexity",
    "potensi_margin",
    "kebutuhan_keahlian",
    "daya_beli_target"
]:
    df[col] = df[col].map(MAP)


# ==================================================
# 4. Feature & Target Selection
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

X = df[FEATURES]
y = df["status_keberhasilan"]


# ==================================================
# 5. Train-Test Split
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples          : {len(X_train)}")
print(f"Testing samples           : {len(X_test)}")


# ==================================================
# 6. Model Definition
# ==================================================
base_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42
)

model = CalibratedClassifierCV(
    estimator=base_model,
    method="sigmoid",
    cv=3
)


# ==================================================
# 7. Training
# ==================================================
model.fit(X_train, y_train)


# ==================================================
# 8. Evaluation
# ==================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    digits=4,
    output_dict=True
)


# ==================================================
# 9. Save Model
# ==================================================
dump(model, MODEL_PATH)


# ==================================================
# 10. Save Training Summary (JSON)
# ==================================================
training_summary = {
    "timestamp": datetime.now().isoformat(),
    "algorithm": "RandomForestClassifier + CalibratedClassifierCV",
    "calibration_method": "sigmoid",
    "features_used": FEATURES,
    "dataset_size": len(df),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "metrics": {
        "accuracy": round(accuracy, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1])
        },
        "classification_report": report
    }
}

with open("models/training_summary.json", "w") as f:
    json.dump(training_summary, f, indent=4)


# ==================================================
# 11. Print Summary
# ==================================================
print("\n=========== TRAINING SUMMARY ===========")
print(f"Algorithm             : {training_summary['algorithm']}")
print(f"Calibration Method    : sigmoid")
print(f"Accuracy              : {accuracy:.4f}")
print(f"ROC-AUC               : {auc:.4f}")
print(f"Model saved to        : {MODEL_PATH}")
print(f"Training summary JSON : models/training_summary.json")

print("\n=========== CONFUSION MATRIX ===========")
print("                Pred 0    Pred 1")
print(f"Actual 0 (Fail)    {cm[0][0]:5d}     {cm[0][1]:5d}")
print(f"Actual 1 (Success) {cm[1][0]:5d}     {cm[1][1]:5d}")

print("\n======= CLASSIFICATION REPORT =======")
print(classification_report(y_test, y_pred, digits=4))

print("====================================================")
print(" TRAINING SELESAI – MODEL CALIBRATED & SIAP DIPAKAI ")
print("====================================================")
