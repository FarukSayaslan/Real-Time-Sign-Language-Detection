import os
import cv2
import numpy as np
import json

# ==========================
# AYARLAR
# ==========================

# Hangi sınıf için veri toplayacaksın? (örn: "4", "A", "B")
LABEL = "1"   # BURAYI HER SEFERİNDE DEĞİŞTİRECEKSİN

IMG_SIZE = 96
DATA_DIR = "asl_dataset_mask"
CALIBRATION_PATH = "calibration.json"

# Klasör yolunu hazırla
out_dir = os.path.join(DATA_DIR, LABEL)
os.makedirs(out_dir, exist_ok=True)

print(f"Kaydedilecek klasör: {out_dir}")

# ==========================
# YARDIMCI FONSİYONLAR
# ==========================
def load_calibration():
    # Varsayılan değerler
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 180, 255], dtype=np.uint8)
    
    if os.path.exists(CALIBRATION_PATH):
        try:
            with open(CALIBRATION_PATH, "r") as f:
                data = json.load(f)
                lower = np.array(data["lower_color"], dtype=np.uint8)
                upper = np.array(data["upper_color"], dtype=np.uint8)
            print("Kalibrasyon dosyadan yüklendi.")
        except Exception as e:
            print(f"Kalibrasyon dosyası okunamadı: {e}. Varsayılan kullanılıyor.")
    else:
        print("Kalibrasyon dosyası bulunamadı. Varsayılanlar kullanılıyor.")
        
    return lower, upper

def save_calibration(lower, upper):
    data = {
        "lower_color": lower.tolist(),
        "upper_color": upper.tolist()
    }
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(data, f)
    print("Kalibrasyon dosyası GÜNCELLENDİ (calibration.json).")

def get_hand_points(x, y, w, h, is_right_hand=True):
    # El iskeleti için bağıl koordinatlar (0-1 arası)
    relative_points = [
        (0.5, 0.7),  # Avuç İçi
        (0.2, 0.5),  # Baş Parmak
        (0.35, 0.3), # İşaret
        (0.5, 0.2),  # Orta
        (0.65, 0.3), # Yüzük
        (0.8, 0.45)  # Serçe
    ]
    
    if not is_right_hand:
        relative_points = [(1.0 - px, py) for px, py in relative_points]
        
    points = []
    for (rx, ry) in relative_points:
        px = int(x + rx * w)
        py = int(y + ry * h)
        points.append((px, py))
        
    return points

def draw_hand_overlay(img, points):
    for (px, py) in points:
        cv2.circle(img, (px, py), 6, (0, 0, 255), 2)
        cv2.circle(img, (px, py), 2, (0, 255, 0), -1)
        
    palm = points[0]
    fingers = points[1:]
    for f in fingers:
        cv2.line(img, palm, f, (0, 255, 255), 1)

def calibrate_color(frame, samples):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H_vals = []
    S_vals = []
    V_vals = []
    
    for (cx, cy) in samples:
        roi_small = hsv_frame[cy-7:cy+7, cx-7:cx+7]
        if roi_small.size == 0: continue
        mean_hsv = np.mean(roi_small, axis=(0, 1))
        H_vals.append(mean_hsv[0])
        S_vals.append(mean_hsv[1])
        V_vals.append(mean_hsv[2])
    
    if not H_vals:
        return None, None

    avg_h = np.mean(H_vals)
    avg_s = np.mean(S_vals)
    avg_v = np.mean(V_vals)
    
    offset_h = 10
    offset_s = 50
    offset_v = 70
    
    lower = np.array([max(0, avg_h - offset_h), max(0, avg_s - offset_s), max(0, avg_v - offset_v)], dtype=np.uint8)
    upper = np.array([min(180, avg_h + offset_h), min(255, avg_s + offset_s), min(255, avg_v + offset_v)], dtype=np.uint8)
    
    return lower, upper

def make_hand_mask(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    mask_3 = cv2.merge([mask, mask, mask])
    return mask_3

# ==========================
# KAMERA
# ==========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

print(f"'{LABEL}' sınıfı için veri toplama başlıyor.")
print("Komutlar:")
print("  h  -> SAĞ/SOL EL değiştir")
print("  c  -> KALİBRE ET (Kaydeder)")
print("  s  -> kare içindeki eli KAYDET")
print("  q  -> çıkış")

img_count = len(os.listdir(out_dir))
print(f"Klasörde zaten {img_count} görüntü var.")

# 1. Kalibrasyon Yükle
lower_color, upper_color = load_calibration()
is_calibrated = (os.path.exists(CALIBRATION_PATH)) 
# Eğer dosya varsa 'calibrated' sayalım, yoksa kullanıcı tekrar yapsın
is_right_hand = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Ortadaki kare (ROI) - SAĞ TARAFA KAYDIRILDI
    size = int(min(h, w) * 0.6)
    cx_roi = int(w * 0.75) 
    cy_roi = h // 2
    
    x1 = max(0, cx_roi - size // 2)
    y1 = max(0, cy_roi - size // 2)
    x2 = min(w, cx_roi + size // 2)
    y2 = min(h, cy_roi + size // 2)

    roi = frame[y1:y2, x1:x2]

    # Ekranda referans için kare çiz
    frame_show = frame.copy()
    cv2.rectangle(frame_show, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Rehber Overlay her zaman gösterilebilir (kalibrasyon yenilemek için)
    # Ama sadece kalibrasyon bozuksa veya kullanıcı isterse
    # Kullanıcı 'c' ye basarsa kalibre ederiz.
    
    hand_points = get_hand_points(x1, y1, x2-x1, y2-y1, is_right_hand)
    
    # Bilgi
    label_info = "Kalibrasyon OK" if is_calibrated else "KALIBRASYON LAZIM (C)"
    color_info = (0, 255, 0) if is_calibrated else (0, 0, 255)
    cv2.putText(frame_show, label_info, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_info, 2)
    
    # Overlay'i sadece kalibrasyon yoksa veya kullanıcı isterse çiziyoruz
    # Burada kullanıcı kolaylık olsun diye hep çizelim ama silik olabilir. 
    # Şimdilik hep çizelim, veri toplarken elin nerede durması gerektiğini gösterir.
    draw_hand_overlay(frame_show, hand_points)

    # ROI'yi maskeye çevir
    if roi.size > 0:
        roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        mask_img = make_hand_mask(roi_resized, lower_color, upper_color)

        # Pencereleri göster
        cv2.imshow("Kamera (referans)", frame_show)
        cv2.imshow(f"Mask ROI - Label: {LABEL}", mask_img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if not is_calibrated:
                print("UYARI: Önce kalibrasyon yapın veya dosyanın yüklendiğinden emin olun!")
            else:
                # Kaydet
                fname = f"{LABEL}_{img_count:04d}.jpg"
                out_path = os.path.join(out_dir, fname)
                cv2.imwrite(out_path, mask_img)
                img_count += 1
                print(f"Kaydedildi: {out_path}")

        elif key == ord('q'):
            break
        elif key == ord('c'):
            new_lower, new_upper = calibrate_color(frame, hand_points)
            if new_lower is not None:
                lower_color = new_lower
                upper_color = new_upper
                is_calibrated = True
                save_calibration(lower_color, upper_color)
                print(f"Kalibre Edildi ve Kaydedildi!")
        elif key == ord('h'):
            is_right_hand = not is_right_hand
            print(f"El değiştirildi: {'Sağ' if is_right_hand else 'Sol'}")

cap.release()
cv2.destroyAllWindows()

print(f"Toplam kaydedilen görüntü sayısı: {img_count}")
print("Bitti.")
