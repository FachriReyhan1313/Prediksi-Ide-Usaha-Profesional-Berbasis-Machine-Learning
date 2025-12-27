import pandas as pd
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
# Main prediction service (FINAL – RANKING FIX)
# ==================================================
def run_prediction(form):

    # ==================================================
    # 1. Modal mapping
    # ==================================================
    modal_key = form.get("modal")

    if modal_key == "5_jt":
        modal = 5_000_000
    elif modal_key == "10_jt":
        modal = 10_000_000
    else:
        modal = 15_000_000

    # ==================================================
    # 2. Load & filter dataset (modal)
    # ==================================================
    df = pd.read_csv(DATASET_PATH)

    df = df[
        (df["modal_min"] <= modal) &
        (df["modal_max"] >= modal)
    ].copy()

    # ==================================================
    # 3. Preprocessing fitur (WAJIB LENGKAP)
    # ==================================================
    for col in [
        "persaingan",
        "complexity",
        "potensi_margin",
        "kebutuhan_keahlian",
        "daya_beli_target"
    ]:
        df[col] = df[col].map(MAP)

    # ==================================================
    # 4. Machine Learning Prediction
    # ==================================================
    X = df[FEATURES]
    df["probabilitas"] = model.predict_proba(X)[:, 1] * 100

    # ==================================================
    # 5. Business Rules
    # ==================================================
    segment_pasar = form.get("segment_pasar")
    tren_bisnis = form.get("tren_bisnis")
    dekat_perumahan = form.get("dekat_perumahan")
    dekat_sekolah = form.get("dekat_sekolah")

    if segment_pasar:
        df.loc[
            df["sektor"].str.lower() == segment_pasar.lower(),
            "probabilitas"
        ] += 5

    if tren_bisnis:
        df.loc[
            df["sektor"].str.lower() == tren_bisnis.lower(),
            "probabilitas"
        ] += 3

    if dekat_perumahan == "Ya":
        df.loc[df["lokasi_cocok"] == "Perumahan", "probabilitas"] += 5

    if dekat_sekolah == "Ya":
        df.loc[df["lokasi_cocok"] == "Kampus", "probabilitas"] += 5

    df["probabilitas"] = df["probabilitas"].clip(0, 100)

    # ==================================================
    # 6. 🔥 AGGREGATION FIX (ANTI DUPLIKASI USAHA)
    # ==================================================
    df_agg = (
        df
        .sort_values("probabilitas", ascending=False)
        .groupby("nama_usaha", as_index=False)
        .first()
    )

    top_df = df_agg.head(2)

    # ==================================================
    # 7. 🔥 BUILD OUTPUT (RANKING-BASED LABEL)
    # ==================================================
    results = []

    for idx, (_, row) in enumerate(top_df.iterrows()):
        prob = round(row["probabilitas"], 2)

        if idx == 0:
            label = "Sangat Layak Direkomendasikan"
            label_color = "success"   # 🟢 utama
        else:
            label = "Alternatif Usaha Potensial"
            label_color = "warning"   # 🟡 pembanding

        results.append({
            "nama_usaha": row["nama_usaha"],
            "sektor": row["sektor"],
            "probabilitas": prob,
            "label": label,
            "label_color": label_color,
            "alasan": build_reason(row, form, modal)
        })

    return results
