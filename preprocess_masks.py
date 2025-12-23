import os
import cv2
import numpy as np

RAW_DIR = "asl_dataset"        # Şu an kullandığın klasör
OUT_DIR = "asl_dataset_mask"   # Yeni maske datasetimiz

os.makedirs(OUT_DIR, exist_ok=True)

def make_hand_mask(img):
    # İstersen burada resize da yapabilirsin (örn: 96x96)
    # img = cv2.resize(img, (96, 96))

    # 1) BGR -> HSV (ten rengi için daha uygun)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2) Basit bir ten rengi aralığı (ışığa göre değiştirmen gerekebilir)
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 180, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)  # el = beyaz(255), diğer = siyah(0)

    # 3) Gürültüyü azalt (morfolojik işlemler)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # 4) 3 kanallı hale getir (CNN 3 kanal bekliyor)
    mask_3 = cv2.merge([mask, mask, mask])
    return mask_3

for label in os.listdir(RAW_DIR):
    in_class_dir = os.path.join(RAW_DIR, label)
    out_class_dir = os.path.join(OUT_DIR, label)

    if not os.path.isdir(in_class_dir):
        continue

    os.makedirs(out_class_dir, exist_ok=True)

    for fname in os.listdir(in_class_dir):
        in_path = os.path.join(in_class_dir, fname)

        img = cv2.imread(in_path)
        if img is None:
            continue

        mask_img = make_hand_mask(img)

        out_path = os.path.join(out_class_dir, fname)
        cv2.imwrite(out_path, mask_img)

        # İstersen ilerlemesini görmek için:
        # print("Kaydedildi:", out_path)

print("Bitti! Yeni maskele dataset:", OUT_DIR)
