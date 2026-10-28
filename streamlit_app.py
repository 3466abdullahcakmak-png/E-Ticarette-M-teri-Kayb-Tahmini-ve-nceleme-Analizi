"""
================================================================
ADIM 4 — Streamlit Web Arayüzü
Churn analiz sonuçlarını tarayıcıda interaktif dashboard olarak sunar
================================================================
Çalıştırmak için:
    streamlit run streamlit_app.py
================================================================
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_YOLU = os.path.join(BASE_DIR, "Nihai_Karar_Destek_Verisi.csv")

# ── Sayfa yapılandırması ─────────────────────────────────────────
st.set_page_config(
    page_title = "🛒 Churn Analiz Paneli",
    page_icon  = "📊",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Özel CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 10px; color: white;
        text-align: center; margin: 0.3rem;
    }
    .risk-high   { color: #E74C3C; font-weight: bold; }
    .risk-medium { color: #F39C12; font-weight: bold; }
    .risk-low    { color: #27AE60; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Veri yükleme (önbellekli) ───────────────────────────────────
@st.cache_data
def veri_yukle():
    if not os.path.exists(CSV_YOLU):
        st.error(f"CSV bulunamadı: {CSV_YOLU}\nÖnce churn_analizi.py çalıştırın.")
        st.stop()
    return pd.read_csv(CSV_YOLU)


# ════════════════════════════════════════════════════════════════
# SIDEBAR — Filtreler
# ════════════════════════════════════════════════════════════════
def sidebar_filtreler(df):
    st.sidebar.title("🔧 Filtreler")
    st.sidebar.markdown("---")

    risk_secim = st.sidebar.multiselect(
        "Risk Segmenti",
        options=["Yüksek Risk", "Orta Risk", "Düşük Risk"],
        default=["Yüksek Risk", "Orta Risk", "Düşük Risk"]
    )

    kok_neden_secim = st.sidebar.multiselect(
        "Kök Neden",
        options=df["Kok_Neden_Etiketi"].unique().tolist(),
        default=df["Kok_Neden_Etiketi"].unique().tolist()
    )

    min_skor, max_skor = st.sidebar.slider(
        "Churn Risk Skoru Aralığı",
        min_value=0.0, max_value=1.0,
        value=(0.0, 1.0), step=0.01
    )

    st.sidebar.markdown("---")
    st.sidebar.info(f"**Toplam Müşteri:** {len(df):,}")

    filtre = (
        df["Risk_Segmenti"].isin(risk_secim) &
        df["Kok_Neden_Etiketi"].isin(kok_neden_secim) &
        df["Churn_Risk_Skoru"].between(min_skor, max_skor)
    )
    return df[filtre]


# ════════════════════════════════════════════════════════════════
# ANA SAYFA
# ════════════════════════════════════════════════════════════════
def ana_sayfa(df_filtreli, df_tam):

    st.title("🛒 E-Ticaret Churn Analiz Paneli")
    st.caption(f"Son güncelleme: Veri kaynağı → {CSV_YOLU}")
    st.markdown("---")

    # ── KPI Kartları ─────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)

    toplam       = len(df_filtreli)
    yuksek_risk  = (df_filtreli["Risk_Segmenti"] == "Yüksek Risk").sum()
    churn_oran   = df_filtreli["Churn"].mean() * 100
    ort_risk     = df_filtreli["Churn_Risk_Skoru"].mean()
    ort_tenure   = df_filtreli["Tenure"].mean() if "Tenure" in df_filtreli.columns else 0

    k1.metric("👥 Toplam Müşteri",  f"{toplam:,}")
    k2.metric("🔴 Yüksek Risk",     f"{yuksek_risk:,}",
              delta=f"{yuksek_risk/toplam*100:.1f}%")
    k3.metric("📉 Churn Oranı",     f"{churn_oran:.1f}%")
    k4.metric("⚠️ Ort. Risk Skoru", f"{ort_risk:.3f}")
    k5.metric("📅 Ort. Tenure",     f"{ort_tenure:.1f} ay")

    st.markdown("---")

    # ── Grafikler (Satır 1) ──────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Segmenti Dağılımı")
        seg_count = df_filtreli["Risk_Segmenti"].value_counts()
        fig_pie = px.pie(
            values=seg_count.values,
            names=seg_count.index,
            color=seg_count.index,
            color_discrete_map={
                "Yüksek Risk": "#E74C3C",
                "Orta Risk"  : "#F39C12",
                "Düşük Risk" : "#27AE60"
            },
            hole=0.4
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Churn Risk Skoru Dağılımı")
        fig_hist = px.histogram(
            df_filtreli, x="Churn_Risk_Skoru",
            color="Risk_Segmenti",
            color_discrete_map={
                "Yüksek Risk": "#E74C3C",
                "Orta Risk"  : "#F39C12",
                "Düşük Risk" : "#27AE60"
            },
            nbins=40, opacity=0.8
        )
        fig_hist.add_vline(x=0.4, line_dash="dash", line_color="#F39C12",
                           annotation_text="Orta Risk Eşiği")
        fig_hist.add_vline(x=0.7, line_dash="dash", line_color="#E74C3C",
                           annotation_text="Yüksek Risk Eşiği")
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Grafikler (Satır 2) ──────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Kök Neden Analizi")
        kok_count = df_filtreli["Kok_Neden_Etiketi"].value_counts()
        fig_bar = px.bar(
            x=kok_count.values, y=kok_count.index,
            orientation="h",
            color=kok_count.index,
            color_discrete_sequence=["#E74C3C", "#F39C12", "#BDC3C7"],
            labels={"x": "Müşteri Sayısı", "y": ""}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col4:
        if "Tenure" in df_filtreli.columns:
            st.subheader("Tenure vs Churn Risk Skoru")
            fig_scatter = px.scatter(
                df_filtreli,
                x="Tenure", y="Churn_Risk_Skoru",
                color="Risk_Segmenti",
                color_discrete_map={
                    "Yüksek Risk": "#E74C3C",
                    "Orta Risk"  : "#F39C12",
                    "Düşük Risk" : "#27AE60"
                },
                opacity=0.6, size_max=8,
                hover_data=["CustomerID", "Kok_Neden_Etiketi"]
            )
            fig_scatter.add_hline(y=0.7, line_dash="dash", line_color="#E74C3C")
            fig_scatter.add_hline(y=0.4, line_dash="dash", line_color="#F39C12")
            st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # ── Yüksek Risk Müşteri Tablosu ──────────────────────────────
    st.subheader("🔴 Yüksek Riskli Müşteriler — Öncelik Listesi")

    goster_sutunlar = [
        "CustomerID", "Churn_Risk_Skoru", "Risk_Segmenti",
        "Kok_Neden_Etiketi", "Tenure", "CashbackAmount"
    ]
    mevcut = [c for c in goster_sutunlar if c in df_filtreli.columns]

    tablo_df = (
        df_filtreli[df_filtreli["Risk_Segmenti"] == "Yüksek Risk"]
        [mevcut]
        .sort_values("Churn_Risk_Skoru", ascending=False)
        .head(50)
    )

    def renk_risk(val):
        if isinstance(val, float):
            if val >= 0.7:
                return "color: #E74C3C; font-weight: bold"
            elif val >= 0.4:
                return "color: #F39C12"
        return ""

    st.dataframe(
        tablo_df.style.applymap(renk_risk, subset=["Churn_Risk_Skoru"]),
        use_container_width=True, height=400
    )

    # ── CSV İndirme ──────────────────────────────────────────────
    st.download_button(
        label     = "📥 Filtrelenmiş Veriyi İndir (CSV)",
        data      = df_filtreli.to_csv(index=False, encoding="utf-8-sig"),
        file_name = "filtrelenmis_churn_data.csv",
        mime      = "text/csv"
    )


# ════════════════════════════════════════════════════════════════
# GİRİŞ NOKTASI
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df_tam = veri_yukle()
    df_filtreli = sidebar_filtreler(df_tam)
    ana_sayfa(df_filtreli, df_tam)
