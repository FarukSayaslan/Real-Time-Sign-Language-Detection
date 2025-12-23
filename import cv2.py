import cv2

for i in range(6):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Kamera {i} AÇILDI")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Kamera {i}", frame)
            cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"Kamera {i} AÇILMADI")
