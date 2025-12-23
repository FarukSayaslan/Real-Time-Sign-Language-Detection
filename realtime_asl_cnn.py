import cv2
import numpy as np
import tensorflow as tf
import json
import time
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "asl_cnn_best.h5"
CLASS_INDICES_PATH = "class_indices.json"
CALIBRATION_PATH = "calibration.json"
IMG_SIZE = 96
CONF_THRESHOLD = 0.50

# Cümle Kurma Ayarları
STABILITY_THRESHOLD = 15
SPACE_THRESHOLD = 40

# ==========================
#  YARDIMCI FONKSIYONLAR
# ==========================
def nothing(x):
    pass

def save_calibration(lower, upper):
    data = {
        "lower_color": lower.tolist(),
        "upper_color": upper.tolist()
    }
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(data, f)
    print(f"Kalibrasyon KAYDEDİLDİ: {CALIBRATION_PATH}")

def load_calibration():
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
            print(f"Kalibrasyon yüklenemedi: {e}")
    else:
        print("Kalibrasyon dosyası yok, varsayılanlar kullanılıyor.")
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
#  MAIN LOOP
# ==========================
def main():
    # 1. Model Yükle
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model yüklendi.")
        with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
            class_indices = json.load(f)
        idx_to_label = {int(v): k for k, v in class_indices.items()}
    except Exception as e:
        print("HATA: Model/JSON bulunamadı!", e)
        return

    # 2. Kalibrasyon Yükle
    lower_color, upper_color = load_calibration()

    # 3. Ayar Penceresi (Trackbars)
    cv2.namedWindow("Ayarlar")
    cv2.resizeWindow("Ayarlar", 400, 300)
    cv2.createTrackbar("H Min", "Ayarlar", lower_color[0], 180, nothing)
    cv2.createTrackbar("H Max", "Ayarlar", upper_color[0], 180, nothing)
    cv2.createTrackbar("S Min", "Ayarlar", lower_color[1], 255, nothing)
    cv2.createTrackbar("S Max", "Ayarlar", upper_color[1], 255, nothing)
    cv2.createTrackbar("V Min", "Ayarlar", lower_color[2], 255, nothing)
    cv2.createTrackbar("V Max", "Ayarlar", upper_color[2], 255, nothing)

    # 4. Kamera Başlat
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera hatası!")
        return

    print("ALGILAMA BAŞLADI.")
    print("Q: Çıkış | R: Temizle | B: Sil")
    print("V: Görünüm Değiştir (Normal/Maske/Overlay)")
    print("S: Ayarları Kaydet")

    prev_frame_time = 0
    
    # Logic Değişkenleri
    current_sentence = ""
    last_pred_char = None
    stable_frame_count = 0
    no_hand_frame_count = 0
    
    # Görünüm: 0=Normal, 1=Maske, 2=Overlay
    view_mode = 0 
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time

        # Trackbar Değerlerini Oku (Anlık Güncelleme)
        h_min = cv2.getTrackbarPos("H Min", "Ayarlar")
        s_min = cv2.getTrackbarPos("S Min", "Ayarlar")
        v_min = cv2.getTrackbarPos("V Min", "Ayarlar")
        h_max = cv2.getTrackbarPos("H Max", "Ayarlar")
        s_max = cv2.getTrackbarPos("S Max", "Ayarlar")
        v_max = cv2.getTrackbarPos("V Max", "Ayarlar")
        
        lower_color = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper_color = np.array([h_max, s_max, v_max], dtype=np.uint8)

        # ROI
        size = int(min(h, w) * 0.6)
        cx_roi = int(w * 0.75) 
        cy_roi = h // 2
        x1, y1 = max(0, cx_roi - size // 2), max(0, cy_roi - size // 2)
        x2, y2 = min(w, cx_roi + size // 2), min(h, cy_roi + size // 2)

        roi = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        predicted_char = "?"
        confidence = 0.0
        mask_display = None
        
        if roi.size > 0:
            roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            mask_img = make_hand_mask(roi_resized, lower_color, upper_color)
            mask_display = cv2.resize(mask_img, (x2-x1, y2-y1)) # Gösterim için büyüt

            # Doluluk Oranı Kontrolü
            white_pixels = np.count_nonzero(mask_img)
            fill_ratio = white_pixels / mask_img.size
            
            if fill_ratio > 0.05: 
                no_hand_frame_count = 0
                
                img_input = mask_img.astype("float32")
                img_input = preprocess_input(img_input)
                img_input = np.expand_dims(img_input, axis=0)

                preds = model.predict(img_input, verbose=0)[0]
                class_id = int(np.argmax(preds))
                confidence = float(np.max(preds))

                if confidence >= CONF_THRESHOLD:
                    predicted_char = idx_to_label.get(class_id, "?")
            else:
                no_hand_frame_count += 1

        # Cümle Mantığı
        if predicted_char != "?" and predicted_char == last_pred_char:
            stable_frame_count += 1
            if stable_frame_count == STABILITY_THRESHOLD:
                current_sentence += predicted_char
        else:
            stable_frame_count = 0
            last_pred_char = predicted_char
            
        if no_hand_frame_count == SPACE_THRESHOLD:
            if current_sentence and not current_sentence.endswith(" "):
                current_sentence += " "
            no_hand_frame_count += 1
            
        # Görünüm İşleme
        frame_show = frame.copy()
        
        if view_mode == 1 and mask_display is not None:
            # Maske Modu (Sadece ROI bölgesini maske yap, gerisi dursun veya full maske?)
            # Kullanıcı "elimi siyah beyaz göreyim" dedi. ROI kısmını değiştirelim.
            frame_show[y1:y2, x1:x2] = mask_display
            cv2.putText(frame_show, "Gorunum: MASKE (v)", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
        elif view_mode == 2 and mask_display is not None:
            # Overlay Modu (Maskeyi kırmızı bindir)
            # Maskeyi BGR'ye çevirip kırmızı yapalım (B=0, G=0, R=Mask) - Basitçe addWeighted
            # mask_display zaten BGR dönüşlü (make_hand_mask 3 kanal döndürüyor)
            # Ama o grayscale'in 3 kanallısı (beyaz). 
            # Kırmızı yapmak için:
            mask_single = cv2.cvtColor(mask_display, cv2.COLOR_BGR2GRAY)
            zero_channel = np.zeros_like(mask_single)
            mask_red = cv2.merge([zero_channel, zero_channel, mask_single]) # Sadece R kanalı dolu (Mavi-Yeşil-Kırmızı) -> BGR: (0,0,255) Kırmızı
            
            roi_bg = frame[y1:y2, x1:x2]
            combined = cv2.addWeighted(roi_bg, 0.7, mask_red, 0.5, 0)
            frame_show[y1:y2, x1:x2] = combined
            cv2.putText(frame_show, "Gorunum: OVERLAY (v)", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        else:
            cv2.putText(frame_show, "Gorunum: NORMAL (v)", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # UI Metinleri
        cv2.putText(frame_show, f"Tahmin: {predicted_char} ({confidence:.2f})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame_show, f"FPS: {int(fps)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Alt Panel
        cv2.rectangle(frame_show, (0, h-60), (w, h), (0, 0, 0), -1)
        cv2.putText(frame_show, f"Yazi: {current_sentence}", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("ASL Proje - Algilama", frame_show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            current_sentence = ""
        elif key == ord('b'):
            current_sentence = current_sentence[:-1]
        elif key == ord('v'):
            view_mode = (view_mode + 1) % 3
        elif key == ord('s'):
            save_calibration(lower_color, upper_color)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
