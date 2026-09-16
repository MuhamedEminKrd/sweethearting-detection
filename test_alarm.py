import cv2
import sys
import numpy as np
import time
from ultralytics import YOLO

def calculate_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def point_in_poly(pt, poly):
    if len(poly) < 3:
        return False
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

def select_polygon(frame):
    points = []
    window_name = "Alan Secimi (Nokta Nokta) - Bitirmek icin ENTER"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)
    
    def draw_polygon(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
    
    cv2.setMouseCallback(window_name, draw_polygon)
    
    print("[BILGI] Kasa/Etkileşim bölgesini NOKTA NOKTA (farenin sol tuşuyla tıklayarak) seçin.")
    print("[BILGI] Seçimi bitirmek için ENTER veya SPACE tuşuna basın.")
    
    while True:
        display = frame.copy()
        if len(points) > 0:
            cv2.polylines(display, [np.array(points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            for p in points:
                cv2.circle(display, p, 5, (0, 0, 255), -1)
                
        cv2.putText(display, "Nokta Koymak Icin Tiklayin. Bitirmek Icin ENTER", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 32: # ENTER or SPACE
            break
            
    cv2.destroyWindow(window_name)
    return np.array(points, np.int32)

def main():
    video_path = "hirsiz.mp4"
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

    # 1. KULLANICIYA KASA BÖLGESİNİ POLİGON OLARAK SEÇTİR
    ret, first_frame = cap.read()
    if not ret:
        print("[HATA] İlk kare okunamadı.")
        sys.exit(1)
        
    KASA_ZONE_POLY = select_polygon(first_frame)
    if len(KASA_ZONE_POLY) < 3:
        print("[HATA] En az 3 nokta seçmelisiniz. Çıkılıyor.")
        sys.exit(1)

    # Durum (State) Takibi
    state = {
        "kasa_visited": False,
        "kasa_time": 0.0,
        "alarm_active": False,
        "alarm_time": 0.0
    }

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    out = cv2.VideoWriter('alarm_test_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    show_ui = len(sys.argv) == 1
    if show_ui:
        cv2.namedWindow("Sweethearting Alarm Testi", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Sweethearting Alarm Testi", 960, 720)

    frame_count = 0
    print("[BILGI] Analiz başladı...")

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        current_time = time.time()
        
        results = model(frame, conf=0.3, verbose=False)
        
        # Kasa Bölgesini Çiz (Poligon olarak)
        cv2.polylines(frame, [KASA_ZONE_POLY], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # Yarı saydam iç dolgu
        overlay = frame.copy()
        cv2.fillPoly(overlay, [KASA_ZONE_POLY], (255, 0, 0))
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Poly'nin merkezini bulup yazı yazdırmak için
        M = cv2.moments(KASA_ZONE_POLY)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(frame, "KASA/ETKILESIM", (cX - 50, cY), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if state["alarm_active"] and current_time - state["alarm_time"] > 4.0:
            state["alarm_active"] = False

        if state["kasa_visited"] and current_time - state["kasa_time"] > 8.0:
            state["kasa_visited"] = False

        debug_text = []

        if results and len(results[0].keypoints) > 0:
            kpts = results[0].keypoints.data[0].cpu().numpy()
            
            # SOL KOL KONTROLÜ
            if kpts[5][2] > 0.3 and kpts[7][2] > 0.3 and kpts[9][2] > 0.3:
                p_sh = (int(kpts[5][0]), int(kpts[5][1]))
                p_el = (int(kpts[7][0]), int(kpts[7][1]))
                p_wr = (int(kpts[9][0]), int(kpts[9][1]))
                
                cv2.line(frame, p_sh, p_el, (0, 255, 255), 3)
                cv2.line(frame, p_el, p_wr, (0, 255, 255), 3)
                cv2.circle(frame, p_sh, 6, (0, 200, 255), -1)
                cv2.circle(frame, p_el, 6, (0, 255, 255), -1)
                cv2.circle(frame, p_wr, 6, (0, 255, 0), -1)
                
                angle = calculate_angle(p_sh, p_el, p_wr)
                cv2.putText(frame, f"{int(angle)} deg", (p_el[0] + 15, p_el[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if point_in_poly(p_wr, KASA_ZONE_POLY):
                    state["kasa_visited"] = True
                    state["kasa_time"] = current_time

                is_wrist_down = p_wr[1] > (p_sh[1] + 30) # Bilek omuzdan biraz aşağıda
                is_wrist_close_to_body = abs(p_wr[0] - p_sh[0]) < 200
                is_arm_bent = angle < 110.0 # Biraz tolerans tanındı

                debug_text.append(f"Aci < 110: {is_arm_bent} ({int(angle)})")
                debug_text.append(f"Bilek Asagida: {is_wrist_down}")
                debug_text.append(f"Bilek Vucuda Yakin: {is_wrist_close_to_body}")

                if state["kasa_visited"] and not state["alarm_active"]:
                    if is_wrist_down and is_wrist_close_to_body and is_arm_bent:
                        state["alarm_active"] = True
                        state["alarm_time"] = current_time
                        state["kasa_visited"] = False 

            # SAĞ KOL KONTROLÜ (Ekstra güvenlik için iki kolu da takip edelim)
            if kpts[6][2] > 0.3 and kpts[8][2] > 0.3 and kpts[10][2] > 0.3:
                p_sh_r = (int(kpts[6][0]), int(kpts[6][1]))
                p_el_r = (int(kpts[8][0]), int(kpts[8][1]))
                p_wr_r = (int(kpts[10][0]), int(kpts[10][1]))
                
                cv2.line(frame, p_sh_r, p_el_r, (0, 140, 255), 3)
                cv2.line(frame, p_el_r, p_wr_r, (0, 140, 255), 3)
                cv2.circle(frame, p_sh_r, 6, (0, 140, 255), -1)
                cv2.circle(frame, p_el_r, 6, (0, 140, 255), -1)
                cv2.circle(frame, p_wr_r, 6, (0, 0, 255), -1)
                
                angle_r = calculate_angle(p_sh_r, p_el_r, p_wr_r)
                cv2.putText(frame, f"{int(angle_r)} deg", (p_el_r[0] - 80, p_el_r[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

                if point_in_poly(p_wr_r, KASA_ZONE_POLY):
                    state["kasa_visited"] = True
                    state["kasa_time"] = current_time

                is_wrist_down_r = p_wr_r[1] > (p_sh_r[1] + 30) 
                is_wrist_close_to_body_r = abs(p_wr_r[0] - p_sh_r[0]) < 200
                is_arm_bent_r = angle_r < 110.0

                if state["kasa_visited"] and not state["alarm_active"]:
                    if is_wrist_down_r and is_wrist_close_to_body_r and is_arm_bent_r:
                        state["alarm_active"] = True
                        state["alarm_time"] = current_time
                        state["kasa_visited"] = False 

        # Ekrana Debug Bilgilerini Yazdır
        y_offset = 60
        for text in debug_text:
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25

        status_text = "DURUM: KASAYA UZANILDI -> CEBE GITMESI BEKLENIYOR" if state["kasa_visited"] else "DURUM: BEKLENIYOR"
        color = (0, 165, 255) if state["kasa_visited"] else (0, 255, 0)
        cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if state["alarm_active"]:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 15)
            cv2.putText(frame, "!!! SUPHELI HAREKET: PARA CEBE ATILDI !!!", (w//2 - 350, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

        out.write(frame)
        if show_ui:
            cv2.imshow("Sweethearting Alarm Testi", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    if show_ui:
        cv2.destroyAllWindows()
    print("[BILGI] Test bitti. Cikti dosyasi: alarm_test_output.mp4")

if __name__ == "__main__":
    main()
