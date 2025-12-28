import pandas as pd
import numpy as np
from config import DATASET_PATH
from models.predictor import model, FEATURES, MAP


# ==================================================
# Helper: Build alasan rekomendasi (SAFE)
# ==================================================
def build_reason(row, form, modal):
    reasons = []

    dekat_sekolah = form.get("dekat_sekolah")
    dekat_perumahan = form.get("dekat_perumahan")
    segment_pasar = form.get("segment_pasar")
    tren_bisnis = form.get("tren_bisnis")
    tempat_usaha = form.get("tempat_usaha")

    if row["modal_min"] <= modal <= row["modal_max"]:
        reasons.append("modal sesuai dengan kebutuhan usaha")

    if dekat_sekolah == "Ya" and row["lokasi_cocok"] == "Kampus":
        reasons.append("lokasi dekat sekolah/kampus")

    if dekat_perumahan == "Ya" and row["lokasi_cocok"] == "Perumahan":
        reasons.append("lokasi dekat kawasan perumahan")

    if segment_pasar and row["sektor"].lower() == segment_pasar.lower():
        reasons.append(f"segment pasar {segment_pasar.lower()} sesuai")

    if tren_bisnis and row["sektor"].lower() == tren_bisnis.lower():
        reasons.append(f"tren {tren_bisnis.lower()} sedang diminati")

    if tempat_usaha == "Ya" and row["complexity"] == MAP["Low"]:
        reasons.append("operasional relatif mudah dikelola")

    if not reasons:
        reasons.append("memiliki peluang stabil berdasarkan pola data historis")

    return ", ".join(reasons)


# ==================================================
# Main prediction service (FINAL – FIX ALL)
# ==================================================
def run_prediction(form):

    # ==================================================
    # 1. Modal mapping (EXISTING)
    # ==================================================
    modal_key = form.get("modal")

    if modal_key == "5_jt":
        modal = 5_000_000
    elif modal_key == "10_jt":
        modal = 10_000_000
    else:
        modal = 15_000_000

    # ==================================================
    # 2. Load & filter dataset (EXISTING)
    # ==================================================
    df = pd.read_csv(DATASET_PATH)

    df = df[
        (df["modal_min"] <= modal) &
        (df["modal_max"] >= modal)
    ].copy()

    # ==================================================
    # 3. Preprocessing fitur (EXISTING)
    # ==================================================
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
    # 3.5 🔥 USER CONTEXT INJECTION (NEW – CORE FIX)
    # ==================================================
    dekat_sekolah = form.get("dekat_sekolah")
    dekat_perumahan = form.get("dekat_perumahan")
    preferensi = form.get("preferensi_usaha")

    if dekat_sekolah == "Ya":
        df["tren_pasar"] = df["tren_pasar"] + 0.4

    if dekat_perumahan == "Ya":
        df["daya_beli_target"] = df["daya_beli_target"] + 0.4

    # Clamp biar tetap valid
    df["tren_pasar"] = df["tren_pasar"].clip(1, 3)
    df["daya_beli_target"] = df["daya_beli_target"].clip(1, 3)

    # ==================================================
    # 3.6 🔥 FILTER BY PREFERENSI USAHA (NEW)
    # ==================================================
    if preferensi and preferensi != "Bebas":
        df = df[df["sektor"].str.lower() == preferensi.lower()]

    # ==================================================
    # 4. Machine Learning Prediction (EXISTING)
    # ==================================================
    X = df[FEATURES]
    df["probabilitas"] = model.predict_proba(X)[:, 1] * 100

    # ==================================================
    # 4.5 🔥 SOFT VARIATION (ANTI HASIL SAMA)
    # ==================================================
    rng = np.random.default_rng(abs(hash(str(form))) % (10**6))
    df["probabilitas"] += rng.normal(0, 2.5, size=len(df))

    # ==================================================
    # 5. Business Rules (EXISTING)
    # ==================================================
    if dekat_perumahan == "Ya":
        df.loc[df["lokasi_cocok"] == "Perumahan", "probabilitas"] += 4

    if dekat_sekolah == "Ya":
        df.loc[df["lokasi_cocok"] == "Kampus", "probabilitas"] += 4

    # ==================================================
    # 5.5 🔥 REALISTIC CLAMP (ANTI 100%)
    # ==================================================
    df["probabilitas"] = df["probabilitas"].clip(45, 92)

    # ==================================================
    # 6. AGGREGATION FIX (EXISTING – ANTI DUPLIKASI)
    # ==================================================
    df_agg = (
        df
        .sort_values("probabilitas", ascending=False)
        .groupby("nama_usaha", as_index=False)
        .first()
    )

    top_df = df_agg.head(2)

    # ==================================================
    # 7. Build Output (EXISTING)
    # ==================================================
    results = []
    for idx, row in top_df.iterrows():
        prob = round(row["probabilitas"], 2)

        if idx == 0:
            label = "Sangat Layak Direkomendasikan"
            label_color = "success"
        else:
            label = "Alternatif Usaha Potensial"
            label_color = "warning"

        results.append({
            "nama_usaha": row["nama_usaha"],
            "sektor": row["sektor"],
            "probabilitas": prob,
            "label": label,
            "label_color": label_color,
            "alasan": build_reason(row, form, modal)
        })

    return results
