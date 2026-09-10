import cv2
import time
import numpy as np
from config import settings

class Overlay:
    def __init__(self):
        self.colors = {
            "KRITIK": (0, 0, 255),   # Kirmizi
            "YUKSEK": (0, 165, 255), # Turuncu
            "ORTA":   (0, 255, 255), # Sari
            "BILGI":  (0, 255, 0)    # Yesil
        }
        self.classes_config = settings.get("classes", {})
        
        # Bolgeleri oku
        self.zones = settings.get("zones", {})
        self.polygons = {}
        for zone_name in ["etkilesim", "kasa", "tehlike"]:
            pts = self.zones.get(zone_name, [])
            if len(pts) >= 3:
                self.polygons[zone_name] = np.array(pts, np.int32)

    def draw_zones(self, frame):
        """Kasa, Etkilesim ve Tehlike bolgelerini yari saydam cizer."""
        overlay = frame.copy()
        
        if "etkilesim" in self.polygons:
            cv2.fillPoly(overlay, [self.polygons["etkilesim"]], (0, 255, 255))
        if "kasa" in self.polygons:
            cv2.fillPoly(overlay, [self.polygons["kasa"]], (0, 255, 0))
        if "tehlike" in self.polygons:
            cv2.fillPoly(overlay, [self.polygons["tehlike"]], (0, 0, 255))
            
        # Yari saydamlik efekti
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        return frame

    def draw_detections(self, frame, detections):
        """Eldeki tespitleri (ve ID'lerini) cizer."""
        for det in detections:
            name = det["class_name"]
            conf = det["confidence"]
            track_id = det.get("track_id", -1)
            x1, y1, x2, y2 = det["bbox"]

            severity = self.classes_config.get(name, "BILGI")
            color = self.colors.get(severity, (255, 255, 255))
            thickness = 2
            
            # Kutu ciz
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Merkez noktasini ciz
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)

            # Etiket: eger ID varsa ekle
            label = f"{name} {conf:.2f}"
            if track_id != -1:
                label = f"ID:{track_id} " + label

            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return frame

    def draw_status_bar(self, frame, rule_engine, show_alarm=False):
        """Ekranin en altina durum cubugu cizer."""
        height, width, _ = frame.shape
        cv2.rectangle(frame, (0, height - 30), (width, height), (0, 0, 0), -1)

        # Orta Kisim: ALARM veya Saat
        if show_alarm:
            cv2.putText(frame, "ALARM: HIRSIZLIK SUPHESI - BÖLGE İHLALİ!",
                        (width // 2 - 210, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["KRITIK"], 2)
        else:
            current_time = time.strftime("%H:%M:%S")
            cv2.putText(frame, f"Sistem Aktif - {current_time}", (width // 2 - 100, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Sag Kisim: Cikis kisayolu
        cv2.putText(frame, "[q]=Cikis", (width - 100, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return frame
