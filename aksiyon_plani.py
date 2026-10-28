"""
================================================================
ADIM 5 — Müşteri Aksiyon Planı
Yüksek riskli 768 müşteri için otomatik kampanya segmentasyonu
ve e-posta şablonu üretimi
================================================================
"""

import os
import pandas as pd
from datetime import datetime

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_YOLU  = os.path.join(BASE_DIR, "Nihai_Karar_Destek_Verisi.csv")
AKSIYON_DIR = os.path.join(BASE_DIR, "aksiyon_plani")
os.makedirs(AKSIYON_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════
# ADIM 5A — Müşteri Segmentasyon Kuralları
# ════════════════════════════════════════════════════════════════

def kampanya_belirle(row) -> dict:
    """
    Her müşteri için risk profili ve kök neden bileşimine göre
    en uygun kampanya stratejisini belirler.

    Karar Matrisi
    ─────────────
    Yüksek Risk + Operasyonel Sorun → Öncelikli Müşteri Hizmetleri
    Yüksek Risk + Ürün Sorunu       → Ürün Değiştirme / İndirim
    Yüksek Risk + Belirsiz          → VIP Sadakat Paketi
    Orta Risk                       → Kişiselleştirilmiş Teklif
    """
    risk    = row["Risk_Segmenti"]
    neden   = row["Kok_Neden_Etiketi"]
    cashback = row.get("CashbackAmount", 0)
    tenure  = row.get("Tenure", 0)

    if risk == "Yüksek Risk":
        if neden == "Operasyonel Süreç Hatası":
            return {
                "Kampanya_Kodu"  : "OPS-ÖNCE",
                "Kampanya_Adi"   : "Öncelikli Müşteri Hizmetleri",
                "Aksiyon"        : "Dedike müşteri temsilcisi ata, 24 saat içinde ara",
                "Teklif"         : f"%20 indirim kuponu + ücretsiz kargo (1 ay)",
                "Oncelik"        : 1,
                "Tahmini_Maliyet": 150,
            }
        elif neden == "Kötü Ürün Deneyimi":
            return {
                "Kampanya_Kodu"  : "URN-DEG",
                "Kampanya_Adi"   : "Ürün Memnuniyet Garantisi",
                "Aksiyon"        : "Ücretsiz ürün değişimi teklif et",
                "Teklif"         : f"%25 indirim + ücretsiz iade hakkı (3 ay)",
                "Oncelik"        : 1,
                "Tahmini_Maliyet": 200,
            }
        else:
            sadakat_bonus = min(cashback * 1.5, 100)
            return {
                "Kampanya_Kodu"  : "VIP-SAD",
                "Kampanya_Adi"   : "VIP Sadakat Paketi",
                "Aksiyon"        : "Kişiselleştirilmiş e-posta + özel teklif gönder",
                "Teklif"         : f"{sadakat_bonus:.0f} TL cashback + VIP üyelik (3 ay ücretsiz)",
                "Oncelik"        : 2,
                "Tahmini_Maliyet": 100,
            }
    elif risk == "Orta Risk":
        return {
            "Kampanya_Kodu"  : "ORT-TEK",
            "Kampanya_Adi"   : "Kişiselleştirilmiş Teklif",
            "Aksiyon"        : "Otomatik e-posta kampanyası başlat",
            "Teklif"         : "%10 indirim kuponu + çift puan fırsatı",
            "Oncelik"        : 3,
            "Tahmini_Maliyet": 30,
        }
    else:
        return {
            "Kampanya_Kodu"  : "DUS-BUL",
            "Kampanya_Adi"   : "Standart Bülten",
            "Aksiyon"        : "Aylık bülten ve ürün önerileri gönder",
            "Teklif"         : "Yeni ürün bildirimleri",
            "Oncelik"        : 5,
            "Tahmini_Maliyet": 5,
        }


# ════════════════════════════════════════════════════════════════
# ADIM 5B — E-Posta Şablonu Üretici
# ════════════════════════════════════════════════════════════════

EMAIL_SABLONLARI = {
    "OPS-ÖNCE": """Konu: Merhaba {ad}, Sizin İçin Buradayız 💙

Sayın Müşterimiz,

Yaşadığınız deneyimin farkındayız ve bunu düzeltmek istiyoruz.
Sizin için özel bir destek temsilcisi atadık.

🎁 Size özel: %20 indirim kuponu → KOD: CHURN20
🚚 1 ay boyunca ücretsiz kargo

Bugün sizi aramak için izninizi bekliyoruz.

Saygılarımızla,
Müşteri Mutluluk Ekibi
""",
    "URN-DEG": """Konu: Ürününüzü Ücretsiz Değiştirelim 🎁

Sayın Müşterimiz,

Ürünümüzden tam memnun olmadığınızı biliyoruz.
Memnuniyetiniz bizim için en önemli öncelik.

✅ Ücretsiz ürün değişimi hakkınız var
💰 %25 indirim kuponunuz hazır → KOD: DEGISTIR25
📦 3 ay boyunca ücretsiz iade

Sizi geri kazanmak için elimizden geleni yapacağız.

Sevgilerimizle,
Ürün Kalite Ekibi
""",
    "VIP-SAD": """Konu: Sizi Özledik! VIP Teklifiniz Bekliyor ⭐

Sayın Değerli Müşterimiz,

{ay} aydır bizimle olduğunuz için teşekkür ederiz.
Sadakatinizi ödüllendirmek istiyoruz.

🌟 VIP üyeliğiniz 3 ay ücretsiz uzatıldı
💵 {cashback} TL cashback hesabınıza tanımlandı
🎯 Size özel ürün önerileri için giriş yapın

Her zaman yanınızdayız.

Saygılarımızla,
VIP Üye İlişkileri
""",
    "ORT-TEK": """Konu: Size Özel %10 İndirim Fırsatı 🛍️

Sayın Müşterimiz,

Alışveriş deneyiminizi daha iyi hale getirmek istiyoruz.

🏷️ %10 indirim kuponunuz: KOD: HOSGELDIN10
⭐ Bu hafta çift puan fırsatı

Alışverişin tadını çıkarın!

Teşekkürler,
Kampanya Ekibi
""",
    "DUS-BUL": """Konu: Bu Haftanın Öne Çıkanları 📬

Merhaba,

Bu hafta platformumuzda neler var bir göz atın!
Size özel ürün önerileri hazırladık.

İyi alışverişler!

Ekibimiz
""",
}


def email_olustur(row) -> str:
    sablon = EMAIL_SABLONLARI.get(row["Kampanya_Kodu"], "")
    return sablon.format(
        ad      = f"Müşteri #{row['CustomerID']}",
        ay      = int(row.get("Tenure", 0)),
        cashback= f"{row.get('CashbackAmount', 0):.0f}"
    )


# ════════════════════════════════════════════════════════════════
# ANA AKIŞ
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("MÜŞTERİ AKSİYON PLANI OLUŞTURULUYOR")
    print("=" * 55)

    df = pd.read_csv(CSV_YOLU)
    print(f"✅ {len(df)} müşteri yüklendi.")

    # ── Kampanya bilgilerini hesapla ─────────────────────────────
    kampanya_df = df.apply(kampanya_belirle, axis=1, result_type="expand")
    df = pd.concat([df, kampanya_df], axis=1)

    # ── E-posta içeriklerini üret ────────────────────────────────
    df["Email_Icerigi"] = df.apply(email_olustur, axis=1)

    # ── Yüksek riskli müşterileri ayır ve sırala ────────────────
    yuksek_risk = (
        df[df["Risk_Segmenti"] == "Yüksek Risk"]
        .sort_values("Churn_Risk_Skoru", ascending=False)
        .copy()
    )
    print(f"\n🔴 Yüksek Riskli Müşteri Sayısı: {len(yuksek_risk)}")

    # ── Kampanya özet raporu ─────────────────────────────────────
    print("\n📊 Kampanya Dağılımı:")
    print(df.groupby("Kampanya_Adi")["CustomerID"].count()
            .sort_values(ascending=False).to_string())

    # ── Tahmini maliyet ─────────────────────────────────────────
    toplam_maliyet = df["Tahmini_Maliyet"].sum()
    print(f"\n💰 Toplam Tahmini Kampanya Maliyeti: {toplam_maliyet:,} TL")

    # ── CSV çıktıları ────────────────────────────────────────────
    # 1. Tüm müşteriler aksiyon planıyla
    tam_yol = os.path.join(AKSIYON_DIR, "Tum_Musteriler_Aksiyon.csv")
    df.drop(columns=["Email_Icerigi"]).to_csv(
        tam_yol, index=False, encoding="utf-8-sig")
    print(f"\n  💾 {tam_yol}")

    # 2. Yüksek risk — öncelikli liste (sadece işlem sütunları)
    oncelik_sutunlar = [
        "CustomerID", "Churn_Risk_Skoru", "Risk_Segmenti",
        "Kok_Neden_Etiketi", "Kampanya_Kodu", "Kampanya_Adi",
        "Aksiyon", "Teklif", "Oncelik", "Tahmini_Maliyet"
    ]
    oncelik_yol = os.path.join(AKSIYON_DIR, "Yuksek_Risk_Oncelik_Listesi.csv")
    yuksek_risk[[c for c in oncelik_sutunlar if c in yuksek_risk.columns]].to_csv(
        oncelik_yol, index=False, encoding="utf-8-sig")
    print(f"  💾 {oncelik_yol}")

    # 3. E-posta listesi (ilk 50 müşteri için örnek)
    email_yol = os.path.join(AKSIYON_DIR, "Email_Ornekleri.txt")
    with open(email_yol, "w", encoding="utf-8") as f:
        f.write(f"E-POSTA ÖRNEKLERİ — {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write("=" * 60 + "\n\n")
        for _, row in yuksek_risk.head(10).iterrows():
            f.write(f"CustomerID: {row['CustomerID']} | "
                    f"Risk: {row['Churn_Risk_Skoru']:.2%} | "
                    f"Kampanya: {row['Kampanya_Kodu']}\n")
            f.write("-" * 60 + "\n")
            f.write(row["Email_Icerigi"])
            f.write("\n" + "=" * 60 + "\n\n")
    print(f"  💾 {email_yol}")

    print("\n✅ Aksiyon planı tamamlandı!")
    print("=" * 55)
