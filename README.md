# 📊 Müşteri Terk (Churn) Analizi ve Erken Uyarı Karar Destek Sistemi

Bu proje, müşteri verilerini, şikayet kayıtlarını ve müşteri yorumlarını makine öğrenmesi ve LLM (Büyük Dil Modelleri) desteğiyle analiz ederek **müşteri kaybını (churn) tahmin eden** ve proaktif aksiyon planları sunan uçtan uca bir veri bilimi çözümüdür.

---

## 🚀 Temel Özellikler

- **Veri Entegrasyonu:** Müşteri demografisi, kullanım alışkanlıkları ve şikayet/yorum verilerinin birleştirilmesi.
- **Makine Öğrenmesi Modeli:** XGBoost ve Random Forest algoritmalarıyla yüksek doğrulukta churn tahmini.
- **Açıklanabilir Yapay Zekâ (XAI):** SHAP analizleri ile her müşteri segmentinin neden riskli olduğunun kök neden analizi.
- **LLM Destekli Duygu & Niyet Analizi:** OpenAI/LLM entegrasyonu ile şikayet metinlerinin özetlenmesi ve müşteri kaybı sinyallerinin yakalanması.
- **Aksiyon Planı & Retention:** Yüksek riskli müşteriler için otomatik aksiyon önerileri ve özelleştirilmiş e-posta şablonları.
- **İnteraktif Web Dashboard:** Streamlit tabanlı dinamik gösterge paneli.

---

## 📁 Proje Klasör Yapısı

`	ext
├── aksiyon_plani/             # Çıktı aksiyon önerileri ve e-posta şablonları
├── data/                      # Ham/örnek veri setleri
├── gorseller/                 # Model ve analiz görselleştirme çıktıları
├── aksiyon_plani.py           # Müşteri segmentlerine göre aksiyon stratejileri üreten betik
├── churn_analizi.py           # Veri ön işleme ve makine öğrenmesi ana pipeline'ı
├── gorsellestirme.py          # Analiz ve SHAP grafiklerini oluşturan modül
├── llm_entegrasyon.py         # LLM API entegrasyonu (OpenAI duygu analizi)
├── model_iyilestirme.py       # Hiperparametre optimizasyonu ve model karşılaştırma
├── streamlit_app.py           # İnteraktif web arayüzü
├── requirements.txt           # Gerekli Python kütüphaneleri
└── README.md                  # Proje dokümantasyonu
`

---

## 🛠️ Kurulum ve Çalıştırma

### 1. Kütüphaneleri Yükleyin
`ash
pip install -r requirements.txt
`

### 2. Analiz Pipeline'ını Çalıştırın
`ash
python churn_analizi.py
`

### 3. Streamlit Web Dashboard'unu Başlatın
`ash
streamlit run streamlit_app.py
`

---

## 📈 Örnek Analiz Çıktıları

Proje kapsamında üretilen analiz grafikleri gorseller/ klasöründe yer almaktadır:
- Churn ve Risk Segmentleri Dağılımı
- Özellik Önem Düzeyleri (Feature Importance)
- SHAP Waterfall & Beeswarm Analizleri
- Tenure (Müşteri Süresi) vs. Risk Dağılımı