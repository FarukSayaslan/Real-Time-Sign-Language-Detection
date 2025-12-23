import cv2
import numpy as np
import json
import os

CALIBRATION_FILE = "calibration.json"

# ==========================
#  YARDIMCI FONKSIYONLAR
# ==========================
def save_calibration(lower, upper):
    data = {
        "lower_color": lower.tolist(),
        "upper_color": upper.tolist()
    }
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(data, f)
    print(f"Kalibrasyon KAYDEDİLDİ: {CALIBRATION_FILE}")
    print(f"Lower: {lower}")
    print(f"Upper: {upper}")

def load_calibration():
    # Varsayılanlar
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 180, 255], dtype=np.uint8)
    
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, "r") as f:
                data = json.load(f)
                lower = np.array(data["lower_color"], dtype=np.uint8)
                upper = np.array(data["upper_color"], dtype=np.uint8)
        except:
            pass
    return lower, upper

def nothing(x):
    pass

def get_hand_points(x, y, w, h, is_right_hand=True):
    relative_points = [
        (0.5, 0.7), (0.2, 0.5), (0.35, 0.3), 
        (0.5, 0.2), (0.65, 0.3), (0.8, 0.45)
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
    if len(points) > 1:
        palm = points[0]
        fingers = points[1:]
        for f in fingers:
            cv2.line(img, palm, f, (0, 255, 255), 1)

def calibrate_color_auto(frame, samples):
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
    
    offset_h = 15
    offset_s = 60
    offset_v = 80
    
    lower = np.array([max(0, avg_h - offset_h), max(0, avg_s - offset_s), max(0, avg_v - offset_v)], dtype=np.uint8)
    upper = np.array([min(180, avg_h + offset_h), min(255, avg_s + offset_s), min(255, avg_v + offset_v)], dtype=np.uint8)
    
    return lower, upper

def make_hand_mask(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    return mask

# ==========================
#  MOD 1: OTOMATİK (REHBERLİ)
# ==========================
def run_auto_calibration(cap):
    is_right_hand = True
    print("\n[OTO MOD] Elinizi çizgilere getirin.")
    print("H: El Değiştir | C: Hesapla | Q: İptal")
    
    preview_mode = False
    preview_lower = None
    preview_upper = None

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # ROI
        size = int(min(h, w) * 0.6)
        cx_roi = int(w * 0.75) 
        cy_roi = h // 2
        x1, y1 = max(0, cx_roi - size // 2), max(0, cy_roi - size // 2)
        x2, y2 = min(w, cx_roi + size // 2), min(h, cy_roi + size // 2)
        
        frame_show = frame.copy()
        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if not preview_mode:
            hand_points = get_hand_points(x1, y1, x2-x1, y2-y1, is_right_hand)
            draw_hand_overlay(frame_show, hand_points)
            
            mode_text = "SAG El" if is_right_hand else "SOL El"
            cv2.putText(frame_show, f"{mode_text} ('h' degis)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame_show, "Ayarla -> 'c'", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            # Önizleme
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                roi_rs = cv2.resize(roi, (96, 96))
                mask = make_hand_mask(roi_rs, preview_lower, preview_upper)
                
                # Maskeyi ROI üzerine bindir
                mask_display = cv2.resize(mask, (x2-x1, y2-y1))
                mask_display_bgr = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)
                roi_bg = frame_show[y1:y2, x1:x2]
                combined = cv2.addWeighted(roi_bg, 0.6, mask_display_bgr, 0.4, 0)
                frame_show[y1:y2, x1:x2] = combined
                
                cv2.putText(frame_show, "ONIZLEME", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame_show, "Y: Kaydet | N: Tekrar", (40, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("ASL Kalibrasyon - OTO", frame_show)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'): return False
        
        if not preview_mode:
            if key == ord('h'): is_right_hand = not is_right_hand
            elif key == ord('c'):
                hand_points = get_hand_points(x1, y1, x2-x1, y2-y1, is_right_hand)
                low, upp = calibrate_color_auto(frame, hand_points)
                if low is not None:
                    preview_lower, preview_upper = low, upp
                    preview_mode = True
        else:
            if key == ord('y'):
                save_calibration(preview_lower, preview_upper)
                return True
            elif key == ord('n'):
                preview_mode = False

# ==========================
#  MOD 2: MANUEL (SLIDER)
# ==========================
def run_manual_calibration(cap):
    print("\n[MANUEL MOD] Slider'lar ile ayar yapın.")
    print("V: Görüntü Modu Değiştir (Maske / Renkli)")
    print("S: Kaydet ve Çık | Q: İptal")
    
    cv2.namedWindow("Ayarlar")
    cv2.resizeWindow("Ayarlar", 400, 300)
    
    # Mevcut ayarları yükle
    cur_lower, cur_upper = load_calibration()
    
    # Trackbar'lar
    cv2.createTrackbar("H Min", "Ayarlar", cur_lower[0], 180, nothing)
    cv2.createTrackbar("H Max", "Ayarlar", cur_upper[0], 180, nothing)
    cv2.createTrackbar("S Min", "Ayarlar", cur_lower[1], 255, nothing)
    cv2.createTrackbar("S Max", "Ayarlar", cur_upper[1], 255, nothing)
    cv2.createTrackbar("V Min", "Ayarlar", cur_lower[2], 255, nothing)
    cv2.createTrackbar("V Max", "Ayarlar", cur_upper[2], 255, nothing)
    
    view_mask_only = False # Toggle durumu
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Değerleri oku
        h_min = cv2.getTrackbarPos("H Min", "Ayarlar")
        h_max = cv2.getTrackbarPos("H Max", "Ayarlar")
        s_min = cv2.getTrackbarPos("S Min", "Ayarlar")
        s_max = cv2.getTrackbarPos("S Max", "Ayarlar")
        v_min = cv2.getTrackbarPos("V Min", "Ayarlar")
        v_max = cv2.getTrackbarPos("V Max", "Ayarlar")
        
        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
        
        # ROI (Algılama ile aynı)
        size = int(min(h, w) * 0.6)
        cx_roi = int(w * 0.75) 
        cy_roi = h // 2
        x1, y1 = max(0, cx_roi - size // 2), max(0, cy_roi - size // 2)
        x2, y2 = min(w, cx_roi + size // 2), min(h, cy_roi + size // 2)
        
        roi = frame[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, (96, 96)) # Model boyutu simülasyonu
        
        mask = make_hand_mask(roi_resized, lower, upper)
        mask_display = cv2.resize(mask, (x2-x1, y2-y1)) # ROI boyutuna geri büyüt
        
        frame_show = frame.copy()
        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Görüntüleme Modu
        if view_mask_only:
            # Sadece siyah - beyaz maske (Ekrana sığdır)
            # Güzel görünmesi için maskeyi ana ekrana yerleştiriyoruz ama
            # daha net görmek için bütün ekranı maskeye çevirebiliriz veya ROI'yi büyütebiliriz.
            # Kullanıcı "elimi siyah beyaz göreyim" dedi.
            frame_show[y1:y2, x1:x2] = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)
            cv2.putText(frame_show, "MOD: SADECE MASKE (v)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Maskeyi BGR üstüne bindir
            mask_bgr = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)
            roi_bg = frame_show[y1:y2, x1:x2]
            # Kırmızımsı bir overlay yapalım maske olan yerlere
            # Maskenin beyaz olduğu yerleri al
            combined = cv2.addWeighted(roi_bg, 0.7, mask_bgr, 0.3, 0)
            frame_show[y1:y2, x1:x2] = combined
            cv2.putText(frame_show, "MOD: KARISIK (v)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        cv2.putText(frame_show, "Kaydet: S | Cikis: Q", (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("ASL Kalibrasyon - MANUEL", frame_show)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'): 
            cv2.destroyWindow("Ayarlar")
            return False
        elif key == ord('v'):
            view_mask_only = not view_mask_only
        elif key == ord('s'):
            save_calibration(lower, upper)
            cv2.destroyWindow("Ayarlar")
            return True

# ==========================
#  MAIN
# ==========================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    print("ASL KALİBRASYON")
    print("1. [O]tomatik Mod (Rehberli)")
    print("2. [M]anuel Mod (Slider)")
    print("Seçim yapın (o/m) veya bu pencerede tuşa basın...")
    
    # Seçim ekranı için basit döngü
    selected_mode = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        cv2.putText(frame, "SECIM YAPIN:", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "A: Otomatik (Rehber)", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "M: Manuel (Slider)", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Q: Cikis", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("ASL Kalibrasyon Secim", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == ord('a') or key == ord('o'):
            selected_mode = 'auto'
            break
        elif key == ord('m'):
            selected_mode = 'manual'
            break
            
    cv2.destroyWindow("ASL Kalibrasyon Secim")
    
    if selected_mode == 'auto':
        run_auto_calibration(cap)
    elif selected_mode == 'manual':
        run_manual_calibration(cap)
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
