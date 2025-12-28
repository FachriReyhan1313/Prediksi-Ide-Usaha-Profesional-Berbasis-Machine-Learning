import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import pandas as pd
from datetime import datetime
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from config import DATASET_PATH, MODEL_PATH


print("==============================================")
print(" TRAINING MODEL IDE USAHA (GB + CALIBRATED) ")
print("==============================================")


# ==================================================
# 1. Load dataset
# ==================================================
df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded : {len(df)} records")


# ==================================================
# 2. Encoding
# ==================================================
MAP = {"Low": 1, "Medium": 2, "High": 3}

for col in [
    "persaingan",
    "complexity",
    "potensi_margin",
    "kebutuhan_keahlian",
    "daya_beli_target",
    "repeat_customer_rate",
    "tren_pasar"
]:
    df[col] = df[col].map(MAP)


# ==================================================
# 3. LABEL LOGIC BARU (BALANCE)
# ==================================================
df["status_keberhasilan"] = (
    df["potensi_margin"] * 0.30 +
    df["daya_beli_target"] * 0.25 +
    df["repeat_customer_rate"] * 0.25 +
    df["tren_pasar"] * 0.20
) >= 2.3

df["status_keberhasilan"] = df["status_keberhasilan"].astype(int)


# ==================================================
# 4. Features
# ==================================================
FEATURES = [
    "modal_min",
    "modal_max",
    "persaingan",
    "complexity",
    "potensi_margin",
    "kebutuhan_keahlian",
    "daya_beli_target",
    "repeat_customer_rate",
    "tren_pasar"
]

X = df[FEATURES]
y = df["status_keberhasilan"]


# ==================================================
# 5. Split
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==================================================
# 6. Model (Gradient Boosting)
# ==================================================
base_model = GradientBoostingClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model = CalibratedClassifierCV(
    estimator=base_model,
    method="sigmoid",
    cv=5
)

model.fit(X_train, y_train)


# ==================================================
# 7. Evaluation
# ==================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)


# ==================================================
# 8. Save model
# ==================================================
dump(model, MODEL_PATH)


# ==================================================
# 9. Save summary
# ==================================================
summary = {
    "timestamp": datetime.now().isoformat(),
    "algorithm": "GradientBoosting + Calibration",
    "dataset_size": len(df),
    "accuracy": round(acc, 4),
    "roc_auc": round(auc, 4),
    "confusion_matrix": {
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1]),
    }
}

with open("models/training_summary.json", "w") as f:
    json.dump(summary, f, indent=4)


print("\n========== TRAINING RESULT ==========")
print(f"Accuracy : {acc:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print("Model saved & ready for prediction")
print("====================================")
