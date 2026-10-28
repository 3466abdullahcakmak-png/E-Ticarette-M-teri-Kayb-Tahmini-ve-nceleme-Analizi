"""
================================================================
E-Ticaret Platformu: Müşteri Kaybı (Churn) Tahmini ve
LLM Destekli İnceleme Analizi
================================================================
Yazar      : Antigravity AI
Tarih      : 2026-10-26
Açıklama   : Bu betik; müşteri, şikayet ve yorum verilerini
              birleştirerek churn tahmini yapar ve metin
              belgelerinden kök neden analizi üretir.
              Çıktı Power BI / Tableau için CSV formatındadır.
================================================================
"""

# ── Standart kütüphane imports ──────────────────────────────────
import os
import re
import zipfile
import warnings
import logging

# ── Üçüncü taraf kütüphane imports ──────────────────────────────
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report,
                             roc_auc_score,
                             confusion_matrix)
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Loglama yapılandırması ───────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# BÖLÜM 0 – SABİTLER VE YOLLAR
# ════════════════════════════════════════════════════════════════

# Proje kök dizini (bu betiğin bulunduğu klasör)
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
# ZIP dosyas gerekiyorsa yolu buraya yazlabilir (CSV'ler data/ altnda olduu iin bo braklabilir)
ZIP_PATH  = os.path.join(DATA_DIR, "data.zip")
CIKTI_CSV = os.path.join(BASE_DIR, "Nihai_Karar_Destek_Verisi.csv")

# Eğer dosya başka bir program tarafından açıksa, Masaüstü'ne kaydet
try:
    _test = open(CIKTI_CSV, "a")
    _test.close()
except PermissionError:
    CIKTI_CSV = os.path.join(
        os.path.expanduser("~"), "Desktop", "Nihai_Karar_Destek_Verisi.csv"
    )

# Makine öğrenmesi hiperparametreleri
RANDOM_STATE = 42
TEST_SIZE    = 0.20
N_ESTIMATORS = 300

# OpenAI API anahtarı (gerçek kullanımda ortam değişkeninden okunur)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# ════════════════════════════════════════════════════════════════
# BÖLÜM 1 – VERİ ENTEGRASYONU
# ════════════════════════════════════════════════════════════════

def zip_ac_ve_yukle(zip_yolu: str, hedef_klas: str) -> None:
    """
    Verilen ZIP dosyasını hedef klasöre açar.
    Klasör zaten mevcutsa ve CSV'ler varsa bu adımı atlar.
    """
    csv_listesi = ["Musteriler.csv", "Sikayetler.csv", "Yorumlar.csv"]
    tumu_mevcut = all(
        os.path.exists(os.path.join(hedef_klas, f)) for f in csv_listesi
    )
    if tumu_mevcut:
        log.info("CSV dosyaları zaten mevcut, ZIP açma adımı atlandı.")
        return

    os.makedirs(hedef_klas, exist_ok=True)
    log.info("ZIP açılıyor: %s → %s", zip_yolu, hedef_klas)
    with zipfile.ZipFile(zip_yolu, "r") as zf:
        zf.extractall(hedef_klas)
    log.info("ZIP başarıyla açıldı.")


def verileri_yukle(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    data_dir klasöründeki üç CSV dosyasını okur ve DataFrame olarak döndürür.

    Returns
    -------
    musteriler, sikayetler, yorumlar : pd.DataFrame üçlüsü
    """
    log.info("CSV dosyaları yükleniyor...")
    musteriler  = pd.read_csv(os.path.join(data_dir, "Musteriler.csv"))
    sikayetler  = pd.read_csv(os.path.join(data_dir, "Sikayetler.csv"))
    yorumlar    = pd.read_csv(os.path.join(data_dir, "Yorumlar.csv"))

    log.info("Musteriler  : %d satır, %d sütun", *musteriler.shape)
    log.info("Sikayetler  : %d satır, %d sütun", *sikayetler.shape)
    log.info("Yorumlar    : %d satır, %d sütun", *yorumlar.shape)
    return musteriler, sikayetler, yorumlar


def veri_birlestir(
    musteriler : pd.DataFrame,
    sikayetler : pd.DataFrame,
    yorumlar   : pd.DataFrame
) -> pd.DataFrame:
    """
    Üç veri setini CustomerID üzerinden LEFT JOIN ile birleştirir.

    Neden LEFT JOIN?
    ─────────────────
    Her müşteri kaydının korunması gerekiyor; bazı müşterilerin
    şikayeti veya yorumu olmayabilir. LEFT JOIN, şikayet/yorum
    bilgisi eksik olsa bile müşteri satırını tutar — eksik değerler
    NaN olarak gelir ve ilerleyen adımda doldurulur.

    Şikayet ve Yorum tablolarında bir müşteriye ait birden fazla
    kayıt olabileceğinden, bu tabloları önce müşteri bazında
    birleştiriyoruz (ilk kaydı alıyoruz ya da metinleri concat ediyoruz).
    """
    log.info("Veri setleri birleştiriliyor...")

    # ── Şikayetleri müşteri bazında birleştir ────────────────────
    # Bir müşterinin birden fazla şikayeti olabilir.
    # ComplaintText metinleri " | " ayracıyla birleştirilir;
    # diğer sütunlar için ilk kayıt alınır.
    sik_grp = (
        sikayetler
        .groupby("CustomerID", sort=False)
        .agg(
            ComplaintText     = ("ComplaintText",     lambda x: " | ".join(x.dropna().astype(str))),
            ComplaintCategory = ("ComplaintCategory", "first"),
            ComplaintIntent   = ("ComplaintIntent",   "first")
        )
        .reset_index()
    )

    # ── Yorumları müşteri bazında birleştir ──────────────────────
    # Bir müşterinin birden fazla ürün yorumu olabilir.
    # ReviewText metinleri birleştirilir; Rating ortalaması alınır.
    yor_grp = (
        yorumlar
        .groupby("CustomerID", sort=False)
        .agg(
            ReviewText     = ("ReviewText",     lambda x: " | ".join(x.dropna().astype(str))),
            Rating         = ("Rating",         "mean"),
            DepartmentName = ("DepartmentName", "first")
        )
        .reset_index()
    )

    # ── Birleştirme (LEFT JOIN zinciri) ──────────────────────────
    df = (
        musteriler
        .merge(sik_grp, on="CustomerID", how="left")
        .merge(yor_grp, on="CustomerID", how="left")
    )

    log.info("Birleştirilmiş veri havuzu: %d satır, %d sütun", *df.shape)
    return df


# ════════════════════════════════════════════════════════════════
# BÖLÜM 2 – VERİ ÖN İŞLEME
# ════════════════════════════════════════════════════════════════

def eksik_veri_doldur(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eksik (NaN) değerleri sütun türüne göre uygun istatistiksel
    yöntemle doldurur.

    Strateji
    ────────
    • Sayısal sütunlar → MEDYAN   (aykırı değerlere karşı sağlam)
    • Kategorik sütunlar → MOD    (en sık görülen kategori)
    • Metin sütunları (ComplaintText, ReviewText) → "YOK"
      (bu müşteride ilgili veri bulunmadığını açıkça belirtir)
    """
    log.info("Eksik veriler dolduruluyor...")

    # Metin sütunları önce ayır
    metin_sutunlar = ["ComplaintText", "ReviewText",
                      "ComplaintCategory", "ComplaintIntent", "DepartmentName"]

    for sutun in df.columns:
        eksik_sayisi = df[sutun].isna().sum()
        if eksik_sayisi == 0:
            continue

        if sutun in metin_sutunlar:
            dolgu = "YOK"
        elif df[sutun].dtype in [np.float64, np.int64, float, int]:
            dolgu = df[sutun].median()
        else:
            mod_deger = df[sutun].mode()
            dolgu = mod_deger[0] if not mod_deger.empty else "Bilinmiyor"

        df[sutun] = df[sutun].fillna(dolgu)
        log.info("  %-35s → %d eksik değer dolduruldu (dolgu: %s)",
                 sutun, eksik_sayisi, dolgu)

    return df


def kategorik_encode_et(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Makine öğrenmesi modeli için kategorik (object tipindeki)
    sütunları sayısal değerlere dönüştürür.

    Yöntem: Label Encoding
    ─────────────────────
    Her benzersiz kategori bir tam sayıya eşlenir.
    Encoder nesneleri, ileride yeni verileri dönüştürmek veya
    orijinal etiketlere geri döndürmek için bir sözlükte saklanır.

    Metin sütunları (NLP için kullanılacak) encode EDİLMEZ;
    bunlar analiz aşamasında ham hâlde tutulur.
    """
    log.info("Kategorik değişkenler encode ediliyor...")

    # Encode edilmeyecek sütunlar (metin analizi için ham tutulur)
    encode_disinda = {"CustomerID", "ComplaintText", "ReviewText"}

    encoderlar = {}
    for sutun in df.select_dtypes(include=["object"]).columns:
        if sutun in encode_disinda:
            continue
        le = LabelEncoder()
        df[sutun] = le.fit_transform(df[sutun].astype(str))
        encoderlar[sutun] = le
        log.info("  %-35s → %d benzersiz sınıf encode edildi", sutun, len(le.classes_))

    return df, encoderlar


# ════════════════════════════════════════════════════════════════
# BÖLÜM 3 – MAKİNE ÖĞRENMESİ İLE CHURN TAHMİNİ
# ════════════════════════════════════════════════════════════════

def ozellikleri_hazirla(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Modele girecek özellik matrisini (X) ve hedef değişkeni (y) hazırlar.

    Dışlanan sütunlar
    ─────────────────
    • CustomerID   : kimlik numarası, model için anlamsız
    • Churn        : hedef değişken, X'e dahil edilmez
    • ComplaintText, ReviewText : ham metin (NLP bölümünde kullanılır)
    """
    disla = {"CustomerID", "Churn", "ComplaintText", "ReviewText"}
    ozellik_sutunlar = [c for c in df.columns
                        if c not in disla and df[c].dtype != object]

    X = df[ozellik_sutunlar]
    y = df["Churn"]
    log.info("Özellik matrisi: %d satır × %d sütun", *X.shape)
    return X, y


def model_egit_ve_skorla(
    df    : pd.DataFrame,
    X     : pd.DataFrame,
    y     : pd.Series
) -> tuple[pd.DataFrame, object]:
    """
    XGBoost sınıflandırıcısı ile churn modelini eğitir,
    her müşteri için [0, 1] aralığında bir 'Churn_Risk_Skoru'
    hesaplar ve bu skoru df'e yeni sütun olarak ekler.

    Aynı zamanda yedek olarak Random Forest de eğitir;
    XGBoost başarısız olursa RF'e düşer (try/except).

    Model Seçim Gerekçesi
    ──────────────────────
    XGBoost:
      • Gradient boosting temelli; eksik değerlere karşı dayanıklı
      • Büyük ve gürültülü e-ticaret verilerinde üstün performans
      • Yerleşik feature importance ile yorumlanabilirlik

    Random Forest (yedek):
      • Aşırı öğrenmeye (overfitting) karşı sağlam
      • Hiper-parametre ayarı daha az kritik
    """
    log.info("Veri train/test olarak ayrılıyor (%.0f%% / %.0f%%)...",
             (1 - TEST_SIZE) * 100, TEST_SIZE * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y          # sınıf dağılımını koru
    )

    # ── Model eğitimi ────────────────────────────────────────────
    try:
        log.info("XGBoost modeli eğitiliyor...")
        model = xgb.XGBClassifier(
            n_estimators      = N_ESTIMATORS,
            max_depth         = 6,
            learning_rate     = 0.05,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            use_label_encoder = False,
            eval_metric       = "logloss",
            random_state      = RANDOM_STATE,
            n_jobs            = -1
        )
        model_adi = "XGBoost"
    except Exception as hata:
        log.warning("XGBoost başlatılamadı (%s). Random Forest'e geçiliyor.", hata)
        model = RandomForestClassifier(
            n_estimators = N_ESTIMATORS,
            max_depth    = 8,
            random_state = RANDOM_STATE,
            n_jobs       = -1
        )
        model_adi = "RandomForest"

    model.fit(X_train, y_train)
    log.info("%s modeli eğitildi.", model_adi)

    # ── Test seti değerlendirmesi ────────────────────────────────
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    log.info("\n%s", "=" * 55)
    log.info("MODEL PERFORMANS RAPORU (%s)", model_adi)
    log.info("%s", "=" * 55)
    log.info("\n%s", classification_report(y_test, y_pred,
                                          target_names=["Kalmaya Devam", "Churn"]))
    log.info("ROC-AUC Skoru : %.4f", roc_auc_score(y_test, y_pred_prob))
    cm = confusion_matrix(y_test, y_pred)
    log.info("Karmaşıklık Matrisi:\n%s", cm)
    log.info("%s\n", "=" * 55)

    # ── Tüm veri seti için churn riski hesapla ───────────────────
    # Modeli tüm satırlara uygulayarak her müşteriye bir risk skoru atıyoruz.
    # Bu skor [0, 1] arasındadır: 1'e yaklaştıkça churn olasılığı yüksek.
    df["Churn_Risk_Skoru"] = model.predict_proba(X)[:, 1]
    df["Churn_Risk_Skoru"] = df["Churn_Risk_Skoru"].round(4)

    # ── Risk segmentleri ─────────────────────────────────────────
    def risk_segmenti(skor):
        if skor >= 0.70:
            return "Yüksek Risk"
        elif skor >= 0.40:
            return "Orta Risk"
        else:
            return "Düşük Risk"

    df["Risk_Segmenti"] = df["Churn_Risk_Skoru"].apply(risk_segmenti)

    # ── Özellik önem skoru (yorumlanabilirlik) ───────────────────
    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)
    log.info("En Önemli 10 Özellik:\n%s", importances.head(10).to_string())

    return df, model


# ════════════════════════════════════════════════════════════════
# BÖLÜM 4 – LLM İLE KÖK NEDEN SINIFLANDIRMASI
# ════════════════════════════════════════════════════════════════

# ── 4a: Kural tabanlı simülasyon (LLM API olmadan) ──────────────

# Kök neden etiketleri
KOK_NEDEN_ETIKETLERI = {
    "Operasyonel Süreç Hatası": [
        "cancel", "iptal", "delay", "gecikme", "shipping", "kargo",
        "delivery", "teslimat", "return", "iade", "refund", "geri ödeme",
        "wrong item", "yanlış ürün", "missing", "eksik", "damage", "hasar",
        "lost", "kayıp", "invoice", "fatura", "payment", "ödeme",
        "account", "hesap", "order", "sipariş", "tracking", "takip",
    ],
    "Kötü Ürün Deneyimi": [
        "quality", "kalite", "poor", "kötü", "bad", "defective", "bozuk",
        "fake", "sahte", "size", "beden", "fit", "uymuyor", "color", "renk",
        "fabric", "kumaş", "material", "malzeme", "broken", "kırık",
        "disappointed", "hayal kırıklığı", "not as described", "açıklamaya uygun değil",
        "uncomfortable", "rahatsız", "cheap", "ucuz görünümlü", "ugly", "çirkin",
    ],
}


def kural_bazli_kok_neden(metin: str) -> str:
    """
    Basit anahtar kelime eşleştirmesiyle kök neden etiketi döndürür.
    LLM API bağlantısı yoksa bu fonksiyon devreye girer.

    Algoritma:
    ──────────
    Her etiket grubundaki anahtar kelimelerden kaç tanesi metinde
    geçiyorsa sayılır; en yüksek sayıyı alan etiket seçilir.
    Beraberlik veya sıfır eşleşme durumunda "Belirsiz" döner.
    """
    # float / NaN gelebilir; güvenli dönüşüm
    if not isinstance(metin, str):
        metin = str(metin) if metin is not None else ""
    metin_kucuk = metin.lower()
    skorlar = {}
    for etiket, anahtar_kelimeler in KOK_NEDEN_ETIKETLERI.items():
        skor = sum(1 for kw in anahtar_kelimeler if kw in metin_kucuk)
        skorlar[etiket] = skor

    max_skor = max(skorlar.values())
    if max_skor == 0:
        return "Belirsiz"

    # Beraberlik kontrolü
    en_yuksekler = [e for e, s in skorlar.items() if s == max_skor]
    if len(en_yuksekler) > 1:
        return "Belirsiz"

    return en_yuksekler[0]


# ── 4b: OpenAI API entegrasyon taslağı ──────────────────────────

def llm_kok_neden_api(metin: str, api_anahtari: str) -> str:
    """
    OpenAI Chat Completions API kullanarak kök neden sınıflandırması yapar.

    Parametre
    ─────────
    metin       : Analiz edilecek şikayet veya yorum metni
    api_anahtari: OpenAI API key (OPENAI_API_KEY ortam değişkeni)

    Dönüş
    ─────
    "Operasyonel Süreç Hatası" | "Kötü Ürün Deneyimi" | "Belirsiz"

    Not
    ───
    Bu fonksiyon gerçek API çağrısı yapar; geçerli bir API
    anahtarı olmadan hata fırlatır. Anahtarınız yoksa
    kural_bazli_kok_neden() fonksiyonu otomatik devreye girer.
    """
    try:
        from openai import OpenAI   # openai>=1.0.0
    except ImportError:
        raise ImportError("openai paketi yüklü değil. `pip install openai` çalıştırın.")

    istemci = OpenAI(api_key=api_anahtari)

    sistem_promptu = """Sen bir e-ticaret müşteri deneyimi analistisisin.
Sana verilen müşteri şikayet veya yorum metnini analiz et ve
SADECE aşağıdaki iki etiketten birini döndür:
- "Operasyonel Süreç Hatası"  → kargo, teslimat, iade, ödeme, sipariş gibi süreç sorunları
- "Kötü Ürün Deneyimi"        → ürün kalitesi, beden, malzeme, görünüm gibi ürün odaklı sorunlar
- "Belirsiz"                  → emin olamazsan

Sadece etiketi yaz, başka hiçbir şey ekleme."""

    yanit = istemci.chat.completions.create(
        model       = "gpt-4o-mini",
        temperature = 0,
        max_tokens  = 20,
        messages    = [
            {"role": "system", "content": sistem_promptu},
            {"role": "user",   "content": metin[:1500]}   # token limiti için kırp
        ]
    )

    etiket = yanit.choices[0].message.content.strip()
    # Beklenen etiketlerden biri değilse güvenli dönüş
    gecerli = {"Operasyonel Süreç Hatası", "Kötü Ürün Deneyimi", "Belirsiz"}
    return etiket if etiket in gecerli else "Belirsiz"


def kok_neden_siniflandir(df: pd.DataFrame, api_anahtari: str = "") -> pd.DataFrame:
    """
    Tüm DataFrame üzerinde kök neden sınıflandırmasını çalıştırır.

    Strateji
    ────────
    1. Geçerli bir API anahtarı varsa → OpenAI API kullanılır.
       (Büyük veri setlerinde maliyet kontrolü için önce 50 satır
        API, geri kalanı kural tabanlı olarak işlenebilir.)
    2. API anahtarı yoksa veya çağrı başarısız olursa →
       kural_bazli_kok_neden() devreye girer.

    Analiz için 'ComplaintText' ve 'ReviewText' birleştirilir;
    bu sayede hem şikayet hem de yorum birlikte değerlendirilir.
    """
    log.info("Kök neden sınıflandırması başlıyor...")

    # Şikayet ve yorum metinlerini birleştir
    df["_Birlesik_Metin"] = (
        df["ComplaintText"].astype(str) + " " + df["ReviewText"].astype(str)
    ).str.strip()

    kullan_api = bool(api_anahtari)
    if kullan_api:
        log.info("OpenAI API kullanılacak.")
    else:
        log.info("API anahtarı bulunamadı → Kural tabanlı NLP simülasyonu kullanılıyor.")

    def siniflandir(metin: str) -> str:
        if kullan_api:
            try:
                return llm_kok_neden_api(metin, api_anahtari)
            except Exception as hata:
                log.debug("API hatası (%s) → kural tabanlıya geçildi.", hata)
                return kural_bazli_kok_neden(metin)
        return kural_bazli_kok_neden(metin)

    # tqdm yoksa sade apply kullan
    try:
        from tqdm import tqdm
        tqdm.pandas(desc="Kök neden analizi")
        df["Kok_Neden_Etiketi"] = df["_Birlesik_Metin"].progress_apply(siniflandir)
    except ImportError:
        df["Kok_Neden_Etiketi"] = df["_Birlesik_Metin"].apply(siniflandir)

    # Yardımcı sütunu temizle
    df.drop(columns=["_Birlesik_Metin"], inplace=True)

    dag = df["Kok_Neden_Etiketi"].value_counts()
    log.info("Kök neden dağılımı:\n%s", dag.to_string())

    return df


# ════════════════════════════════════════════════════════════════
# BÖLÜM 5 – ÇIKTI ALMA
# ════════════════════════════════════════════════════════════════

CIKTI_SUTUNLARI = [
    # Kimlik
    "CustomerID",
    # Churn hedef ve skor
    "Churn", "Churn_Risk_Skoru", "Risk_Segmenti",
    # Müşteri demografisi
    "Tenure", "CityTier", "PreferredLoginDevice", "PreferredPaymentMode",
    # Davranışsal özellikler
    "HourSpendOnApp", "NumberOfDeviceRegistered", "NumberOfAddress",
    "OrderCount", "DaySinceLastOrder", "CashbackAmount",
    "OrderAmountHikeFromlastYear", "WarehouseToHome",
    # Şikayet bilgileri
    "Complain", "ComplaintCategory", "ComplaintIntent", "ComplaintText",
    # Yorum bilgileri
    "ReviewText", "Rating", "DepartmentName",
    # NLP çıktısı
    "Kok_Neden_Etiketi",
]


def nihai_csv_olustur(df: pd.DataFrame, cikti_yolu: str) -> None:
    """
    Power BI / Tableau'da doğrudan kullanılabilecek nihai CSV'yi oluşturur.

    Sadece iş zekası için anlamlı olan sütunlar seçilir,
    geri kalan ara hesaplama sütunları dışarıya verilmez.
    """
    # Tabloda mevcut olan sütunları al (bazıları eksik olabilir)
    mevcut = [s for s in CIKTI_SUTUNLARI if s in df.columns]
    df_cikti = df[mevcut].copy()

    df_cikti.to_csv(cikti_yolu, index=False, encoding="utf-8-sig")
    log.info("Nihai CSV oluşturuldu: %s", cikti_yolu)
    log.info("Çıktı boyutu: %d satır × %d sütun", *df_cikti.shape)

    # ── Özet istatistikler ───────────────────────────────────────
    churn_oran = df_cikti["Churn"].mean() * 100
    ort_risk   = df_cikti["Churn_Risk_Skoru"].mean()
    log.info("Gerçek Churn Oranı  : %.1f%%", churn_oran)
    log.info("Ortalama Risk Skoru : %.4f", ort_risk)

    seg_dag = df_cikti["Risk_Segmenti"].value_counts()
    log.info("Risk Segmenti Dağılımı:\n%s", seg_dag.to_string())

    kok_dag = df_cikti["Kok_Neden_Etiketi"].value_counts()
    log.info("Kök Neden Dağılımı:\n%s", kok_dag.to_string())


# ════════════════════════════════════════════════════════════════
# ANA AKIŞ – PIPELINE ORKESTRASYonu
# ════════════════════════════════════════════════════════════════

def pipeline_calistir() -> pd.DataFrame:
    """
    Tüm adımları sırasıyla çalıştıran ana pipeline fonksiyonu.

    Adımlar
    ───────
    1. ZIP'i aç ve CSV'leri yükle
    2. Veri setlerini birleştir
    3. Eksik değerleri doldur + Encode et
    4. ML modeli ile Churn_Risk_Skoru hesapla
    5. LLM / kural tabanlı kök neden analizi
    6. Nihai CSV'yi dışa aktar
    """
    log.info("═" * 55)
    log.info("PIPELINE BAŞLIYOR")
    log.info("═" * 55)

    # ── 1. Veri yükleme ──────────────────────────────────────────
    zip_ac_ve_yukle(ZIP_PATH, DATA_DIR)
    musteriler, sikayetler, yorumlar = verileri_yukle(DATA_DIR)

    # ── 2. Birleştirme ───────────────────────────────────────────
    df = veri_birlestir(musteriler, sikayetler, yorumlar)

    # Ham metin sütunlarını ML öncesi yedekle
    df_metin_yedek = df[["CustomerID", "ComplaintText", "ReviewText"]].copy()

    # ── 3. Ön işleme ─────────────────────────────────────────────
    df = eksik_veri_doldur(df)
    df, encoderlar = kategorik_encode_et(df)

    # ── 4. Makine öğrenmesi ──────────────────────────────────────
    X, y = ozellikleri_hazirla(df)
    df, model = model_egit_ve_skorla(df, X, y)

    # Encode edilmiş metin sütunlarını ham versiyonla geri al
    # (NLP analizinde okunabilir metin gerekiyor)
    df = df.drop(columns=["ComplaintText", "ReviewText"], errors="ignore")
    df = df.merge(df_metin_yedek, on="CustomerID", how="left")

    # ── 5. Kök neden sınıflandırması ─────────────────────────────
    df = kok_neden_siniflandir(df, api_anahtari=OPENAI_API_KEY)

    # ── 6. Çıktı ─────────────────────────────────────────────────
    nihai_csv_olustur(df, CIKTI_CSV)

    log.info("═" * 55)
    log.info("PIPELINE BAŞARIYLA TAMAMLANDI")
    log.info("Çıktı dosyası: %s", CIKTI_CSV)
    log.info("═" * 55)

    return df


# ════════════════════════════════════════════════════════════════
# GİRİŞ NOKTASI
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sonuc_df = pipeline_calistir()

    # İlk 5 satırı ekrana yazdır
    print("\n── Nihai Veri Seti (İlk 5 Satır) ──")
    pd.set_option("display.max_columns", 8)
    pd.set_option("display.width", 120)
    print(sonuc_df[[
        "CustomerID", "Churn", "Churn_Risk_Skoru",
        "Risk_Segmenti", "Kok_Neden_Etiketi"
    ]].head())
