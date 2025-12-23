import cv2

for index in range(3):   # 0,1,2 dene
    print(f"Deniyorum: Kamera index {index}")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Windows için CAP_DSHOW ekledik
    if cap.isOpened():
        print(f"-> Kamera {index} ACILDI!")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Kamera {index}", frame)
            cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"-> Kamera {index} AÇILAMADI.")
