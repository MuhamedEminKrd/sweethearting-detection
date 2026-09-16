import cv2
import sys
import numpy as np
from ultralytics import YOLO

def calculate_angle(a, b, c):
    """
    a: Omuz (x, y)
    b: Dirsek (x, y) - köşe noktası
    c: Bilek (x, y)
    Dirsek eklemindeki açıyı (derece cinsinden) hesaplar.
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
        
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def main():
    video_path = "Ekran Kaydı 2026-09-13 010423.mp4"
    
    print("[BILGI] İskelet modeli yükleniyor (yolov8n-pose.pt)...")
    try:
        model = YOLO("yolov8n-pose.pt")
    except Exception as e:
        print(f"[HATA] Model yüklenemedi: {e}")
        sys.exit(1)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[HATA] Video açılamadı: {video_path}")
        sys.exit(1)

    print("[BILGI] Sadece KOL İSKELETİ gösteriliyor. Çıkmak için 'q' tuşuna basın.")
    
    # Pencereyi ayarla
    window_name = "Kol Iskelet & Dirsek Aci Takibi"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)
    
    while True:
        success, frame = cap.read()
        if not success:
            # Video bitince başa sar
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = model(frame, conf=0.3, verbose=False)
        
        if results and len(results[0].keypoints) > 0:
            for person in results[0].keypoints.data:
                kpts = person.cpu().numpy()
                
                # COCO Noktaları:
                # 5: Sol Omuz, 7: Sol Dirsek, 9: Sol Bilek
                # 6: Sağ Omuz, 8: Sağ Dirsek, 10: Sağ Bilek
                
                # --- SOL KOL (Sarı Çizgiler) ---
                if kpts[5][2] > 0.3 and kpts[7][2] > 0.3 and kpts[9][2] > 0.3:
                    p_sh = (int(kpts[5][0]), int(kpts[5][1]))
                    p_el = (int(kpts[7][0]), int(kpts[7][1]))
                    p_wr = (int(kpts[9][0]), int(kpts[9][1]))
                    
                    # Kemik çizgileri
                    cv2.line(frame, p_sh, p_el, (0, 255, 255), 3)
                    cv2.line(frame, p_el, p_wr, (0, 255, 255), 3)
                    
                    # Eklem noktaları
                    cv2.circle(frame, p_sh, 6, (0, 200, 255), -1)
                    cv2.circle(frame, p_el, 6, (0, 255, 255), -1)
                    cv2.circle(frame, p_wr, 6, (0, 255, 0), -1)
                    
                    # Dirsek açısı hesapla ve yaz
                    angle_left = calculate_angle(p_sh, p_el, p_wr)
                    cv2.putText(frame, f"{int(angle_left)} deg", (p_el[0] + 10, p_el[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # --- SAĞ KOL (Turuncu Çizgiler) ---
                if kpts[6][2] > 0.3 and kpts[8][2] > 0.3 and kpts[10][2] > 0.3:
                    p_sh = (int(kpts[6][0]), int(kpts[6][1]))
                    p_el = (int(kpts[8][0]), int(kpts[8][1]))
                    p_wr = (int(kpts[10][0]), int(kpts[10][1]))
                    
                    # Kemik çizgileri
                    cv2.line(frame, p_sh, p_el, (0, 140, 255), 3)
                    cv2.line(frame, p_el, p_wr, (0, 140, 255), 3)
                    
                    # Eklem noktaları
                    cv2.circle(frame, p_sh, 6, (0, 140, 255), -1)
                    cv2.circle(frame, p_el, 6, (0, 140, 255), -1)
                    cv2.circle(frame, p_wr, 6, (0, 0, 255), -1)
                    
                    # Dirsek açısı hesapla ve yaz
                    angle_right = calculate_angle(p_sh, p_el, p_wr)
                    cv2.putText(frame, f"{int(angle_right)} deg", (p_el[0] - 80, p_el[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

        # Bilgilendirme başlığı
        cv2.putText(frame, "KOL ISKELETI (Omuz -> Dirsek -> Bilek)", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
