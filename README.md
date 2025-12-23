ASL PROJESİ - KULLANIM KILAVUZU
=================================================

1. NASIL BAŞLATILIR?
--------------------
Projeyi başlatmak için şu dosyayı çalıştırın:
  python main.py

Program açıldığında size soracaktır:
"Mevcut kalibrasyonu kullanmak ister misiniz? (y/n)"
- y (yes): Evet, direkt algılamaya geç.
- n (no): Hayır, yeniden kalibrasyon yap.

2. KALİBRASYON (calibration.py)
-------------------------------
Eğer kalibrasyon yapacaksanız iki seçenek sunulur:
[A] OTOMATİK MOD (Rehberli):
  - Elinizi ekrandaki çizgilere (iskelet) oturtun.
  - 'H' tuşu: Sağ/Sol el değiştirir.
  - 'C' tuşu: Rengi hesaplar ve önizlemeyi açar.
    - Önizlemede 'Y' ile kaydet, 'N' ile iptal et.

[M] MANUEL MOD (İnce Ayar):
  - Işık kötüyse bunu kullanın.
  - "Ayarlar" penceresinden H, S, V çubuklarını kaydırarak elinizi "beyaz", arka planı "siyah" yapmaya çalışın.
  - 'V' tuşu: Görünümü değiştirir (Sadece Maske / Karışık).
  - 'S' tuşu: Ayarları kaydeder ve çıkar.

3. ALGILAMA VE CÜMLE KURMA (realtime_asl_cnn.py)
------------------------------------------------
Program kamerayı açıp el hareketlerinizi tanımaya başlar.

KLAVYE KISAYOLLARI:
-------------------
[V] GÖRÜNÜM DEĞİŞTİR: 
    - Normal (Kamera)
    - Maske (Siyah-Beyaz) -> Elinizin net görünüp görünmediğini kontrol etmek için harika.
    - Overlay (Kırmızı Maske)

[S] KAYDET:
    - O anki slider ayarlarını "calibration.json" dosyasına kaydeder.

[R] RESET (TEMİZLE):
    - Yazılan cümleyi tamamen siler.

[B] BACKSPACE (SİL):
    - Son harfi siler.

[Q] ÇIKIŞ:
    - Programı kapatır.

CÜMLE KURMA MANTIĞI:
--------------------
- Bir harfi yaklaşık 0.5 saniye sabit tutarsanız yazıya eklenir.
- Elinizi ekrandan 1-2 saniye çekerseniz otomatik BOŞLUK bırakır.
