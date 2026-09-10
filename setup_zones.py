import cv2
import yaml
import os
import numpy as np
from config import settings, CONFIG_PATH

def save_zones(zones):
    import sys
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    
    if "zones" not in config_data:
        config_data["zones"] = {}
        
    config_data["zones"]["etkilesim"] = zones.get("etkilesim", [])
    config_data["zones"]["kasa"] = zones.get("kasa", [])
    config_data["zones"]["tehlike"] = zones.get("tehlike", [])
    
    # Eger komut satirindan video dosyasi verilmisse config'teki kaynagi da otomatik guncelle
    if len(sys.argv) > 1:
        if "source" not in config_data:
            config_data["source"] = {}
        config_data["source"]["type"] = "video"
        config_data["source"]["video_path"] = sys.argv[1].replace("\\", "/")
        print(f"[BILGI] config.yaml icindeki kaynak da '{sys.argv[1]}' olarak guncellendi.")
    
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, default_flow_style=None, sort_keys=False)
    print("\n[BASARILI] Bolgeler config.yaml dosyasina kaydedildi.")

def get_frame():
    import sys
    if len(sys.argv) > 1:
        video_arg = sys.argv[1]
        print(f"[BILGI] Komut satirindan girilen video aciliyor: {video_arg}")
        cap = cv2.VideoCapture(video_arg)
    else:
        source_type = settings["source"]["type"]
        if source_type == "camera":
            print(f"[BILGI] Kamera aciliyor (Index: {settings['source']['camera_index']})...")
            cap = cv2.VideoCapture(settings["source"]["camera_index"])
        else:
            print(f"[BILGI] Config videosu aciliyor: {settings['source']['video_path']}...")
            cap = cv2.VideoCapture(settings["source"]["video_path"])
        
    if not cap.isOpened():
        print("[HATA] Kamera veya video acilamadi! Lutfen yolu kontrol edin.")
        return None
        
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[HATA] Goruntu karesi okunamadi!")
        return None
    return frame

# Fare olaylarini yakalamak icin global degiskenler
current_polygon = []
drawing = False

def mouse_callback(event, x, y, flags, param):
    global current_polygon, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_polygon.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_polygon) > 0:
            current_polygon.pop() # Geri al

def main():
    global current_polygon
    frame = get_frame()
    if frame is None:
        return
        
    zones = {}
    zone_names = [
        ("etkilesim", "Musteri-Kasiyer Etkilesim Bolgesini cizin (Tezgah ustu). Bitirmek icin 'ENTER' a basin."),
        ("kasa", "Kasa Bolgesini cizin (Paranin girmesi gereken yer). Bitirmek icin 'ENTER' a basin."),
        ("tehlike", "Tehlike Bolgesini cizin (Kasiyerin cebi/beli). Bitirmek icin 'ENTER' a basin.")
    ]
    
    cv2.namedWindow("Bolge Cizimi")
    cv2.setMouseCallback("Bolge Cizimi", mouse_callback)
    
    for zone_id, instruction in zone_names:
        current_polygon = []
        while True:
            temp_frame = frame.copy()
            cv2.putText(temp_frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(temp_frame, "Sol Tik: Nokta ekle | Sag Tik: Geri al | ENTER: Tamamla", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Cizili onceki bolgeleri goster
            if "etkilesim" in zones and len(zones["etkilesim"]) > 2:
                pts = np.array(zones["etkilesim"], np.int32).reshape((-1, 1, 2))
                cv2.polylines(temp_frame, [pts], True, (0, 255, 255), 2)
            if "kasa" in zones and len(zones["kasa"]) > 2:
                pts = np.array(zones["kasa"], np.int32).reshape((-1, 1, 2))
                cv2.polylines(temp_frame, [pts], True, (0, 255, 0), 2)
                
            # Su anki poligonu goster
            if len(current_polygon) > 0:
                for pt in current_polygon:
                    cv2.circle(temp_frame, (pt[0], pt[1]), 4, (0, 0, 255), -1)
                if len(current_polygon) > 1:
                    pts = np.array(current_polygon, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(temp_frame, [pts], False, (0, 0, 255), 2)
                    
            cv2.imshow("Bolge Cizimi", temp_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13: # ENTER tusu
                if len(current_polygon) >= 3:
                    zones[zone_id] = current_polygon.copy()
                    break
                else:
                    print("Lutfen gecerli bir poligon icin en az 3 nokta secin!")
                    
    cv2.destroyAllWindows()
    save_zones(zones)

if __name__ == "__main__":
    main()
