import os
import subprocess
import sys
import time

CALIBRATION_FILE = "calibration.json"
CALIBRATION_SCRIPT = "calibration.py"
DETECTION_SCRIPT = "realtime_asl_cnn.py"

def run_script(script_name):
    """
    Python scriptini subprocess olarak çalıştırır.
    """
    print(f"BAŞLATILIYOR: {script_name}...")
    try:
        # Mevcut python yorumlayıcısını kullan
        subprocess.check_call([sys.executable, script_name])
    except subprocess.CalledProcessError as e:
        print(f"HATA: {script_name} çalışırken bir sorun oluştu.")
        return False
    return True

def main():
    print("ASL PROJESİ YÖNETİCİ ARAYÜZÜ")
    print("===============================")
    
    # 1. Kalibrasyon Dosyası Kontrolü
    should_calibrate = True
    
    if os.path.exists(CALIBRATION_FILE):
        print(f"Mevcut kalibrasyon dosyası bulundu: {CALIBRATION_FILE}")
        while True:
            choice = input("Mevcut kalibrasyonu kullanmak ister misiniz? (y/n): ").strip().lower()
            if choice == 'y':
                should_calibrate = False
                break
            elif choice == 'n':
                should_calibrate = True
                break
            else:
                print("Lütfen 'y' veya 'n' giriniz.")
    
    if should_calibrate:
        print("Kalibrasyon modülü başlatılıyor...")
        
        success = run_script(CALIBRATION_SCRIPT)
        
        if not success:
            print("Kalibrasyon başarısız oldu veya iptal edildi. Program sonlandırılıyor.")
            return

        # Tekrar kontrol et
        if not os.path.exists(CALIBRATION_FILE):
            print("Kalibrasyon dosyası oluşturulmadı. Devam edilemiyor.")
            return
            
        print("Kalibrasyon tamamlandı.")
        time.sleep(1)
        
    else:
        print("Mevcut kalibrasyon kullanılıyor.")
        time.sleep(1)

    # 2. Algılama Modülünü Başlat
    run_script(DETECTION_SCRIPT)

if __name__ == "__main__":
    main()
