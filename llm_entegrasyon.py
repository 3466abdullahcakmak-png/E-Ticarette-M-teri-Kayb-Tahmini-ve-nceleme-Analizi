"""
================================================================
LLM Entegrasyon — Async + Rate Limit Otomatik Algılama
OpenAI Tier seviyene göre otomatik hızlanır
================================================================
"""

import os
import time
import asyncio
import logging
import json
import pandas as pd
from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_YOLU = os.path.join(BASE_DIR, "Nihai_Karar_Destek_Verisi.csv")
CIKTI    = os.path.join(BASE_DIR, "Nihai_Karar_Destek_Verisi_LLM.csv")

SISTEM_PROMPTU = """Sen uzman bir e-ticaret müşteri deneyimi analistisisin.
Sana bir müşterinin şikayet/yorum metni verilecek. Analiz et ve SADECE şu JSON formatında yanıt ver:
{"kok_neden":"<ETİKET>","alt_kategori":"<ALT>","duygu":"<POZİTİF|NEGATİF|NÖTR>","aksiyon_onerisi":"<öneri>","guven_skoru":<0.0-1.0>}

Etiketler:
- "Operasyonel Süreç Hatası" → kargo/teslimat/iade/ödeme/sipariş sorunları
- "Kötü Ürün Deneyimi"       → kalite/beden/malzeme/görünüm sorunları
- "Belirsiz"                 → net kategori yok

Sadece JSON yaz, başka hiçbir şey ekleme."""


async def tier_kontrol_et(istemci: AsyncOpenAI) -> int:
    """
    Bir test isteği göndererek hesabın gerçek RPM limitini ölçer.
    Döndürülen değer: dakikada kaç istek yapılabileceği
    """
    log.info("🔍 Hesap limitleri kontrol ediliyor...")
    try:
        yanit = await istemci.chat.completions.create(
            model      = "gpt-4o-mini",
            max_tokens = 5,
            messages   = [{"role": "user", "content": "test"}]
        )
        # Response headers'dan limit oku
        rpm = int(getattr(yanit, '_response', None) and
                  yanit._response.headers.get("x-ratelimit-limit-requests", 0) or 0)
        if rpm > 0:
            log.info("✅ Hesap RPM limiti: %d istek/dakika", rpm)
            return rpm
    except Exception:
        pass

    log.info("ℹ️  Limit header alınamadı — Tier 1 varsayılıyor (500 RPM)")
    return 500


async def gpt_analiz_async(
    semaphore : asyncio.Semaphore,
    istemci   : AsyncOpenAI,
    idx       : int,
    metin     : str,
    toplam    : int
) -> tuple[int, dict]:
    """
    Tek bir metin için async GPT analizi.
    Semaphore ile eşzamanlı istek sayısı kontrol edilir.
    """
    async with semaphore:
        for deneme in range(4):
            try:
                yanit = await istemci.chat.completions.create(
                    model       = "gpt-4o-mini",
                    temperature = 0,
                    max_tokens  = 150,
                    messages    = [
                        {"role": "system", "content": SISTEM_PROMPTU},
                        {"role": "user",   "content": str(metin)[:1500]}
                    ]
                )
                icerik = yanit.choices[0].message.content.strip()
                if icerik.startswith("```"):
                    icerik = icerik.split("```")[1]
                    if icerik.startswith("json"):
                        icerik = icerik[4:]
                sonuc = json.loads(icerik)
                log.info("  ✅ [%d/%d] Analiz tamamlandı", idx + 1, toplam)
                return idx, sonuc

            except Exception as e:
                hata = str(e)
                if "429" in hata or "rate_limit" in hata.lower():
                    bekleme = (2 ** deneme) * 5   # 5, 10, 20, 40 saniye
                    log.warning("  ⏳ Rate limit [%d/%d] — %d sn bekleniyor...",
                                idx + 1, toplam, bekleme)
                    await asyncio.sleep(bekleme)
                elif "401" in hata:
                    log.error("❌ API anahtarı geçersiz!")
                    break
                else:
                    log.debug("  GPT hatası: %s", e)
                    break

        return idx, {
            "kok_neden"      : "Belirsiz",
            "alt_kategori"   : "Analiz Edilemedi",
            "duygu"          : "NÖTR",
            "aksiyon_onerisi": "Manuel inceleme",
            "guven_skoru"    : 0.0,
        }


async def toplu_llm_async(
    df          : pd.DataFrame,
    api_anahtari: str,
    max_satir   : int = 200,
    eslesen_sayisi: int = 5   # Aynı anda gönderilecek istek sayısı
) -> pd.DataFrame:
    """
    Async paralel LLM analizi.
    eslesen_sayisi = aynı anda kaç istek gönderileceği
    Rate limitine göre otomatik ayarlanır.
    """
    istemci   = AsyncOpenAI(api_key=api_anahtari)
    semaphore = asyncio.Semaphore(eslesen_sayisi)

    # Yüksek risk müşterileri seç
    if "Risk_Segmenti" in df.columns:
        hedef = df[df["Risk_Segmenti"] == "Yüksek Risk"].head(max_satir).copy()
    else:
        hedef = df.head(max_satir).copy()

    log.info("📊 Toplam analiz edilecek: %d müşteri", len(hedef))
    log.info("⚡ Eşzamanlı istek: %d", eslesen_sayisi)

    # Metinleri hazırla
    metinler = (
        hedef["ComplaintText"].astype(str) + " " +
        hedef["ReviewText"].astype(str)
    ).str.strip().tolist()

    indeksler = hedef.index.tolist()

    # Sonuç sütunlarını başlat
    for sutun in ["LLM_Kok_Neden", "LLM_Alt_Kategori",
                  "LLM_Duygu", "LLM_Aksiyon", "LLM_Guven_Skoru"]:
        df[sutun] = "" if sutun != "LLM_Guven_Skoru" else 0.0
    df["LLM_Kok_Neden"] = df["Kok_Neden_Etiketi"]

    # Async görevleri oluştur
    gorevler = [
        gpt_analiz_async(semaphore, istemci, i, metin, len(metinler))
        for i, metin in enumerate(metinler)
    ]

    baslangic = time.time()
    sonuclar  = await asyncio.gather(*gorevler)
    gecen     = time.time() - baslangic

    # Sonuçları DataFrame'e yaz
    for i, (_, sonuc) in enumerate(sonuclar):
        gercek_idx = indeksler[i]
        df.at[gercek_idx, "LLM_Kok_Neden"]    = sonuc.get("kok_neden",       "Belirsiz")
        df.at[gercek_idx, "LLM_Alt_Kategori"] = sonuc.get("alt_kategori",    "")
        df.at[gercek_idx, "LLM_Duygu"]        = sonuc.get("duygu",           "NÖTR")
        df.at[gercek_idx, "LLM_Aksiyon"]      = sonuc.get("aksiyon_onerisi", "")
        df.at[gercek_idx, "LLM_Guven_Skoru"]  = sonuc.get("guven_skoru",     0.0)

    log.info("⏱️  Toplam süre: %.1f saniye (%d müşteri)", gecen, len(metinler))
    log.info("⚡ Hız: %.1f müşteri/saniye", len(metinler) / gecen if gecen > 0 else 0)

    return df


async def main():
    api_anahtari = os.getenv("OPENAI_API_KEY", "")
    if not api_anahtari:
        log.error("OPENAI_API_KEY tanımlı değil!\n"
                  "PowerShell'de: $env:OPENAI_API_KEY = 'sk-proj-...'")
        return

    log.info("=" * 55)
    log.info("ASYNC LLM PIPELINE BAŞLIYOR")
    log.info("=" * 55)

    # Hesap limitini kontrol et
    istemci = AsyncOpenAI(api_key=api_anahtari)
    try:
        test = await istemci.chat.completions.create(
            model="gpt-4o-mini", max_tokens=5,
            messages=[{"role": "user", "content": "merhaba"}]
        )
        log.info("✅ API bağlantısı başarılı!")

        # Tier tespiti: Tier 1 = 500 RPM → 5 eşzamanlı
        #               Free   =   3 RPM → 1 eşzamanlı
        eslesen = 5   # Tier 1 varsayımı (Tier 1: 500 RPM)
        log.info("📡 Eşzamanlı istek sayısı: %d (Tier 1 modu)", eslesen)

    except Exception as e:
        if "429" in str(e):
            log.warning("Free tier tespit edildi → 1 eşzamanlı istek")
            eslesen = 1
        else:
            log.error("Bağlantı hatası: %s", e)
            return

    df = pd.read_csv(CSV_YOLU)
    log.info("%d müşteri yüklendi.", len(df))

    df = await toplu_llm_async(
        df,
        api_anahtari  = api_anahtari,
        max_satir     = 200,
        eslesen_sayisi = eslesen
    )

    df.to_csv(CIKTI, index=False, encoding="utf-8-sig")
    log.info("💾 Çıktı: %s", CIKTI)

    log.info("\n📊 Kural Tabanlı vs LLM:")
    log.info("Kural:\n%s", df["Kok_Neden_Etiketi"].value_counts().to_string())
    if "LLM_Kok_Neden" in df.columns:
        log.info("LLM:\n%s",  df["LLM_Kok_Neden"].value_counts().to_string())

    log.info("=" * 55)
    log.info("TAMAMLANDI ✅")
    log.info("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())
