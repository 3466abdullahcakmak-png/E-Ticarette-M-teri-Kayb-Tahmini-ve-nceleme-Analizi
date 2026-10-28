"""
================================================================
ADIM 3 — Model İyileştirme
SHAP değerleri ile model açıklanabilirliği +
Hiperparametre optimizasyonu (GridSearchCV)
================================================================
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
GORSEL_DIR = os.path.join(BASE_DIR, "gorseller")
os.makedirs(GORSEL_DIR, exist_ok=True)


# ── Veriyi hazırla ───────────────────────────────────────────────
def veri_hazirla():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "Musteriler.csv"))

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        mod = df[col].mode()
        df[col] = df[col].fillna(mod[0] if not mod.empty else "Bilinmiyor")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=["CustomerID", "Churn"])
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ════════════════════════════════════════════════════════════════
# ADIM 3A — Hiperparametre Optimizasyonu
# ════════════════════════════════════════════════════════════════
def hiperparametre_optimizasyonu(X_train, y_train):
    """
    GridSearchCV ile en iyi XGBoost parametrelerini bulur.
    5-katlı Stratified Cross-Validation kullanılır.
    ROC-AUC skoru optimize edilir.
    """
    print("\n🔍 Hiperparametre optimizasyonu başlıyor...")
    print("   (Bu işlem birkaç dakika sürebilir...)")

    param_grid = {
        "n_estimators" : [200, 300, 400],
        "max_depth"    : [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample"    : [0.8, 1.0],
    }

    base_model = xgb.XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        base_model, param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"\n✅ En iyi parametreler: {grid_search.best_params_}")
    print(f"   En iyi CV ROC-AUC   : {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_params_


# ════════════════════════════════════════════════════════════════
# ADIM 3B — SHAP Açıklanabilirlik Analizi
# ════════════════════════════════════════════════════════════════
def shap_analizi(model, X_train, X_test, ozellik_adlari):
    """
    SHAP (SHapley Additive exPlanations) değerleri hesaplar.
    Her özelliğin her müşteri için churn kararına ne kadar
    katkıda bulunduğunu gösterir.

    SHAP Nedir?
    ───────────
    Oyun teorisinden türetilen bir yöntem.
    "Bu özellik olmasa model ne kadar farklı karar verirdi?"
    sorusunu matematiksel olarak yanıtlar.
    """
    try:
        import shap
    except ImportError:
        print("⚠️  SHAP yüklü değil. Yükleniyor...")
        os.system("python -m pip install shap --quiet")
        import shap

    print("\n🔬 SHAP değerleri hesaplanıyor...")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # ── SHAP Özet Grafiği (Bar) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=ozellik_adlari,
        plot_type="bar",
        show=False,
        color="#E74C3C"
    )
    plt.title("SHAP — Ortalama Özellik Etkisi (|SHAP Değeri|)",
              fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(GORSEL_DIR, "08_shap_bar.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 {os.path.join(GORSEL_DIR, '08_shap_bar.png')}")

    # ── SHAP Beeswarm Grafiği ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=ozellik_adlari,
        show=False
    )
    plt.title("SHAP — Özellik Etkisi ve Yönü (Beeswarm)",
              fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(GORSEL_DIR, "09_shap_beeswarm.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 {os.path.join(GORSEL_DIR, '09_shap_beeswarm.png')}")

    # ── En kritik müşteri örneği (Waterfall) ────────────────────
    # Test setindeki en yüksek churn riskli müşteri
    tahminler = model.predict_proba(X_test)[:, 1]
    en_riskli = np.argmax(tahminler)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values       = shap_values[en_riskli],
            base_values  = explainer.expected_value,
            data         = X_test.iloc[en_riskli],
            feature_names= ozellik_adlari
        ),
        show=False
    )
    plt.title(f"SHAP Waterfall — En Riskli Müşteri (Risk: {tahminler[en_riskli]:.2%})",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(GORSEL_DIR, "10_shap_waterfall.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 {os.path.join(GORSEL_DIR, '10_shap_waterfall.png')}")

    return shap_values


# ════════════════════════════════════════════════════════════════
# ANA AKIŞ
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("MODEL İYİLEŞTİRME PIPELINE BAŞLIYOR")
    print("=" * 55)

    X_train, X_test, y_train, y_test = veri_hazirla()

    # Hiperparametre optimizasyonu
    en_iyi_model, en_iyi_params = hiperparametre_optimizasyonu(X_train, y_train)

    # Test performansı
    y_prob = en_iyi_model.predict_proba(X_test)[:, 1]
    y_pred = en_iyi_model.predict(X_test)

    print("\n📊 OPTİMİZE EDİLMİŞ MODEL PERFORMANSI:")
    print("=" * 55)
    print(classification_report(y_test, y_pred,
                                target_names=["Kalmaya Devam", "Churn"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # SHAP analizi
    shap_analizi(en_iyi_model, X_train, X_test,
                 list(X_train.columns))

    print(f"\n✅ Model iyileştirme tamamlandı → {GORSEL_DIR}")
    print("=" * 55)
