"""
Iskelet takibi gorsel test scripti.

Amac: yolo26x-pose modelini canli kamerada calistirip, kamera yeni acidan
(kafa - bel alti gorunecek sekilde) neyi ne kadar iyi yakaladigini gozle
gormek. Henuz hicbir kural/karar mantigi yok - sadece iskeleti ciziyor.

Ekranda gorecegin:
    - Her eklem noktasi (burun, goz, kulak, omuz, dirsek, bilek, kalca,
      diz, ayak bilegi) uzerinde nokta
    - Noktalari birbirine baglayan cizgiler (iskelet)
    - Sag ustte, o anki omuz/dirsek/bilek/kalca koordinatlari ve
      'gorunurluk' (visibility) skorlari terminale yazdirilir

Kullanim: python pose_test.py  ('q' ile cik)
"""
import cv2
from ultralytics import YOLO
from config import settings

# COCO 17-keypoint index sozlugu (yolo26/yolov8-pose ile ayni siralama)
KEYPOINT_NAMES = [
    "burun", "sol_goz", "sag_goz", "sol_kulak", "sag_kulak",
    "sol_omuz", "sag_omuz", "sol_dirsek", "sag_dirsek",
    "sol_bilek", "sag_bilek", "sol_kalca", "sag_kalca",
    "sol_diz", "sag_diz", "sol_ayak_bilegi", "sag_ayak_bilegi",
]

# Takip etmek istedigimiz asil noktalar (kural motoru asamasinda kullanilacak)
ILGI_NOKTALARI = ["sol_omuz", "sag_omuz", "sol_dirsek", "sag_dirsek",
                  "sol_bilek", "sag_bilek", "sol_kalca", "sag_kalca"]

print("Model yukleniyor (yolo26x-pose.pt)...")
model = YOLO("yolo26x-pose.pt")

source_type = settings["source"]["type"]
cap = cv2.VideoCapture(settings["source"]["camera_index"] if source_type == "camera" else settings["source"]["video_path"])

if not cap.isOpened():
    print("Kamera/video acilamadi!")
    raise SystemExit(1)

frame_no = 0
while True:
    ok, frame = cap.read()
    if not ok:
        print("Kare okunamadi.")
        break

    results = model.predict(source=frame, verbose=False)
    annotated = results[0].plot()  # iskeleti hazir cizdiriyoruz

    # Terminale her 15 karede bir ilgi noktalarinin koordinat+gorunurlugunu yaz
    frame_no += 1
    if frame_no % 15 == 0 and len(results[0].keypoints) > 0:
        kpts = results[0].keypoints.data[0]  # ilk kisi icin (17,3) tensor: x,y,conf
        print(f"--- Kare {frame_no} ---")
        for name in ILGI_NOKTALARI:
            idx = KEYPOINT_NAMES.index(name)
            x, y, v = kpts[idx].tolist()
            print(f"  {name:15s}: x={x:6.1f} y={y:6.1f} gorunurluk={v:.2f}")

    cv2.imshow("Iskelet Testi (yolo26x-pose)", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
