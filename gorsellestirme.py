"""
================================================================
ADIM 1 — Python ile Görselleştirme
Churn dağılımı, risk segmentleri, özellik önem grafikleri
================================================================
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # GUI gerektirmeden PNG kaydet
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Sabitler ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_YOLU   = os.path.join(BASE_DIR, "Nihai_Karar_Destek_Verisi.csv")
GORSEL_DIR = os.path.join(BASE_DIR, "gorseller")
os.makedirs(GORSEL_DIR, exist_ok=True)

# ── Renk paleti ──────────────────────────────────────────────────
RENKLER = {
    "Yüksek Risk" : "#E74C3C",
    "Orta Risk"   : "#F39C12",
    "Düşük Risk"  : "#27AE60",
    "Churn"       : "#E74C3C",
    "Kalmaya"     : "#2980B9",
    "arka"        : "#F8F9FA",
    "metin"       : "#2C3E50",
}

plt.rcParams.update({
    "figure.facecolor" : RENKLER["arka"],
    "axes.facecolor"   : RENKLER["arka"],
    "font.family"      : "DejaVu Sans",
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
})


def df_yukle() -> pd.DataFrame:
    df = pd.read_csv(CSV_YOLU)
    print(f"✅ CSV yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    return df


# ════════════════════════════════════════════════════════════════
# GRAFİK 1 — Churn Dağılımı (Pasta + Bar yan yana)
# ════════════════════════════════════════════════════════════════
def grafik_churn_dagilimi(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Müşteri Churn Dağılımı", fontsize=15, fontweight="bold",
                 color=RENKLER["metin"])

    counts  = df["Churn"].value_counts()
    etiket  = ["Kalmaya Devam", "Churn"]
    renkler = [RENKLER["Kalmaya"], RENKLER["Churn"]]

    # Pasta grafiği
    axes[0].pie(
        counts, labels=etiket, colors=renkler,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11}
    )
    axes[0].set_title("Oran Dağılımı")

    # Bar grafiği
    bars = axes[1].bar(etiket, counts.values, color=renkler,
                       edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars, counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 30, str(val),
                     ha="center", va="bottom", fontweight="bold", fontsize=12)
    axes[1].set_ylabel("Müşteri Sayısı")
    axes[1].set_title("Mutlak Sayılar")
    axes[1].set_ylim(0, counts.max() * 1.15)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    yol = os.path.join(GORSEL_DIR, "01_churn_dagilimi.png")
    plt.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 {yol}")


# ════════════════════════════════════════════════════════════════
# GRAFİK 2 — Risk Segmenti Dağılımı (Yatay Bar)
# ════════════════════════════════════════════════════════════════
def grafik_risk_segmenti(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Müşteri Risk Segmentleri", fontsize=15, fontweight="bold",
                 color=RENKLER["metin"])

    sira    = ["Yüksek Risk", "Orta Risk", "Düşük Risk"]
    counts  = df["Risk_Segmenti"].value_counts().reindex(sira)
    renkler = [RENKLER[s] for s in sira]

    bars = ax.barh(sira, counts.values, color=renkler,
                   edgecolor="white", linewidth=1.5, height=0.5)
    for bar, val in zip(bars, counts.values):
        pct = val / len(df) * 100
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({pct:.1f}%)", va="center", fontsize=11)

    ax.set_xlabel("Müşteri Sayısı")
    ax.set_xlim(0, counts.max() * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    yol = os.path.join(GORSEL_DIR, "02_risk_segmentleri.png")
    plt.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 {yol}")


# ════════════════════════════════════════════════════════════════
# GRAFİK 3 — Churn Risk Skoru Dağılımı (Histogram + KDE)
# ════════════════════════════════════════════════════════════════
def grafik_risk_skoru_dagilimi(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Churn Risk Skoru Dağılımı", fontsize=15, fontweight="bold",
                 color=RENKLER["metin"])

    churn_yok = df[df["Churn"] == 0]["Churn_Risk_Skoru"]
    churn_var = df[df["Churn"] == 1]["Churn_Risk_Skoru"]

    ax.hist(churn_yok, bins=40, alpha=0.6, color=RENKLER["Kalmaya"],
            label="Kalmaya Devam", edgecolor="white")
    ax.hist(churn_var, bins=40, alpha=0.6, color=RENKLER["Churn"],
            label="Churn", edgecolor="white")

    # Eşik çizgileri
    ax.axvline(0.40, color=RENKLER["Orta Risk"],   linestyle="--",
               linewidth=1.5, label="Orta Risk Eşiği (0.40)")
    ax.axvline(0.70, color=RENKLER["Yüksek Risk"], linestyle="--",
               linewidth=1.5, label="Yüksek Risk Eşiği (0.70)")

    ax.set_xlabel("Churn Risk Skoru")
    ax.set_ylabel("Müşteri Sayısı")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    yol = os.path.join(GORSEL_DIR, "03_risk_skoru_dagilimi.png")
    plt.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 {yol}")


# ════════════════════════════════════════════════════════════════
# GRAFİK 4 — En Önemli 10 Özellik (Feature Importance)
# ════════════════════════════════════════════════════════════════
def grafik_feature_importance(df: pd.DataFrame):
    """
    Churn modelini yeniden eğiterek feature importance alır.
    """
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    # Ham veriyi yeniden yükle ve encode et
    ham_df = pd.read_csv(os.path.join(BASE_DIR, "data", "Musteriler.csv"))

    # Eksik doldur
    for col in ham_df.select_dtypes(include="number").columns:
        ham_df[col] = ham_df[col].fillna(ham_df[col].median())
    for col in ham_df.select_dtypes(include="object").columns:
        mod = ham_df[col].mode()
        ham_df[col] = ham_df[col].fillna(mod[0] if not mod.empty else "Bilinmiyor")
        le = LabelEncoder()
        ham_df[col] = le.fit_transform(ham_df[col].astype(str))

    X = ham_df.drop(columns=["CustomerID", "Churn"])
    y = ham_df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = xgb.XGBClassifier(n_estimators=300, max_depth=6,
                               learning_rate=0.05, random_state=42,
                               eval_metric="logloss", n_jobs=-1)
    model.fit(X_train, y_train)

    importance = pd.Series(model.feature_importances_,
                           index=X.columns).sort_values()

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Churn Tahminine En Çok Katkı Sağlayan Özellikler",
                 fontsize=15, fontweight="bold", color=RENKLER["metin"])

    renkler_bar = [RENKLER["Churn"] if v > importance.quantile(0.7)
                   else RENKLER["Kalmaya"] for v in importance.values]

    bars = ax.barh(importance.index, importance.values,
                   color=renkler_bar, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, importance.values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.set_xlabel("Önem Skoru (Gain)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, importance.max() * 1.2)

    yuksek = mpatches.Patch(color=RENKLER["Churn"],   label="Yüksek Etki")
    dusuk  = mpatches.Patch(color=RENKLER["Kalmaya"], label="Düşük Etki")
    ax.legend(handles=[yuksek, dusuk], loc="lower right", fontsize=10)

    plt.tight_layout()
    yol = os.path.join(GORSEL_DIR, "04_feature_importance.png")
    plt.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 {yol}")


# ════════════════════════════════════════════════════════════════
# GRAFİK 5 — Kök Neden Dağılımı (Donut)
# ════════════════════════════════════════════════════════════════
def grafik_kok_neden(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Kök Neden Analizi", fontsize=15, fontweight="bold",
                 color=RENKLER["metin"])

    # Sol: Tüm müşteriler
    counts = df["Kok_Neden_Etiketi"].value_counts()
    renkler = ["#E74C3C", "#F39C12", "#BDC3C7"]
    wedges, texts, autotexts = axes[0].pie(
        counts, labels=counts.index, colors=renkler,
        autopct="%1.1f%%", startangle=90, pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.6},
        textprops={"fontsize": 10}
    )
    axes[0].set_title("Tüm Müşteriler")

    # Sağ: Sadece Churn olanlar
    churn_df = df[df["Churn"] == 1]
    counts2  = churn_df["Kok_Neden_Etiketi"].value_counts()
    axes[1].pie(
        counts2, labels=counts2.index, colors=renkler,
        autopct="%1.1f%%", startangle=90, pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.6},
        textprops={"fontsize": 10}
    )
    axes[1].set_title("Sadece Churn Olan Müşteriler")

    plt.tight_layout()
    yol = os.path.join(GORSEL_DIR, "05_kok_neden.png")
    plt.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 {yol}")


# ════════════════════════════════════════════════════════════════
# GRAFİK 6 — Tenure vs Churn Risk (Scatter)
# ════════════════════════════════════════════════════════════════
def grafik_tenure_risk(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Tenure (Müşteri Yaşı) ile Churn Riski İlişkisi",
                 fontsize=15, fontweight="bold", color=RENKLER["metin"])

    # Renkleri Risk Segmentine göre ata
    renk_map = {
        "Yüksek Risk" : RENKLER["Yüksek Risk"],
        "Orta Risk"   : RENKLER["Orta Risk"],
        "Düşük Risk"  : RENKLER["Düşük Risk"],
    }
    for segment, renk in renk_map.items():
        alt = df[df["Risk_Segmenti"] == segment]
        ax.scatter(alt["Tenure"], alt["Churn_Risk_Skoru"],
                   c=renk, label=segment, alpha=0.5, s=20, edgecolors="none")

    ax.axhline(0.70, color=RENKLER["Yüksek Risk"], linestyle="--",
               linewidth=1, alpha=0.7)
    ax.axhline(0.40, color=RENKLER["Orta Risk"], linestyle="--",
               linewidth=1, alpha=0.7)
    ax.set_xlabel("Tenure (Ay)")
    ax.set_ylabel("Churn Risk Skoru")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    yol = os.path.join(GORSEL_DIR, "06_tenure_vs_risk.png")
    plt.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 {yol}")


# ════════════════════════════════════════════════════════════════
# GRAFİK 7 — Korelasyon Isı Haritası
# ════════════════════════════════════════════════════════════════
def grafik_korelasyon(df: pd.DataFrame):
    sayisal = df.select_dtypes(include="number").drop(
        columns=["CustomerID"], errors="ignore"
    )
    corr = sayisal.corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    fig.suptitle("Özellikler Arası Korelasyon Isı Haritası",
                 fontsize=15, fontweight="bold", color=RENKLER["metin"])

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, linewidths=0.5,
        annot_kws={"size": 8}, ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    yol = os.path.join(GORSEL_DIR, "07_korelasyon_haritasi.png")
    plt.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 {yol}")


# ════════════════════════════════════════════════════════════════
# ANA AKIŞ
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("GÖRSELLEŞTİRME PIPELINE BAŞLIYOR")
    print("=" * 55)

    df = df_yukle()

    print("\n📊 Grafikler oluşturuluyor...")
    grafik_churn_dagilimi(df)
    grafik_risk_segmenti(df)
    grafik_risk_skoru_dagilimi(df)
    grafik_feature_importance(df)
    grafik_kok_neden(df)
    grafik_tenure_risk(df)
    grafik_korelasyon(df)

    print(f"\n✅ 7 grafik oluşturuldu → {GORSEL_DIR}")
    print("=" * 55)
