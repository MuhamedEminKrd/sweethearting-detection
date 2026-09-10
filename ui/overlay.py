import cv2
import time
from config import settings

class Overlay:
    def __init__(self):
        # Renk paletimiz (OpenCV BGR formatı kullanır, RGB değil)
        self.colors = {
            "KRITIK": (0, 0, 255),   # Kırmızı
            "YUKSEK": (0, 165, 255), # Turuncu
            "ORTA":   (0, 255, 255), # Sarı
            "BILGI":  (0, 255, 0)    # Yeşil
        }
        self.classes_config = settings["classes"]

    SKELETON_BAGLANTILARI = [
        ("sol_omuz", "sag_omuz"),
        ("sol_omuz", "sol_dirsek"), ("sol_dirsek", "sol_bilek"),
        ("sag_omuz", "sag_dirsek"), ("sag_dirsek", "sag_bilek"),
        ("sol_omuz", "sol_kalca"), ("sag_omuz", "sag_kalca"),
        ("sol_kalca", "sag_kalca"),
    ]

    def draw_pose(self, frame, people, alarm_taraflari=None):
        """
        Omuz/dirsek/bilek/kalca iskeletini cizer. Alarm veren taraf (sol_bilek/
        sag_bilek) varsa o noktayi kirmizi ile vurgular.
        """
        alarm_taraflari = alarm_taraflari or set()

        for person in people:
            for a, b in self.SKELETON_BAGLANTILARI:
                pa, pb = person[a], person[b]
                cv2.line(frame, (int(pa["x"]), int(pa["y"])), (int(pb["x"]), int(pb["y"])), (0, 200, 0), 2)

            for name in ("sol_omuz", "sag_omuz", "sol_dirsek", "sag_dirsek",
                         "sol_bilek", "sag_bilek", "sol_kalca", "sag_kalca"):
                p = person[name]
                color = (0, 0, 255) if name in alarm_taraflari else (0, 200, 0)
                cv2.circle(frame, (int(p["x"]), int(p["y"])), 5, color, -1)

        return frame

    def draw_detections(self, frame, detections):
        """
        Gelen el tespitlerini görüntünün üzerine kutu olarak çizer.
        """
        for det in detections:
            name = det["class_name"]
            conf = det["confidence"]
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]

            # config.yaml dosyasından bu nesnenin önem derecesini (KRITIK, BILGI vs) al
            severity = self.classes_config.get(name, "BILGI")
            color = self.colors.get(severity, (255, 255, 255))
            thickness = 3 if severity == "KRITIK" else (2 if severity in ["YUKSEK", "ORTA"] else 1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            label = f"{name} {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return frame

    def draw_status_bar(self, frame, rule_engine, show_alarm=False):
        """
        Ekranın en altına kasa (POS) durumunu, cooldown sayacını, alarm uyarısını
        ve tuş kısayollarını gösteren bilgi çubuğunu çizer.
        """
        height, width, _ = frame.shape

        # En alta siyah bir şerit çek
        cv2.rectangle(frame, (0, height - 30), (width, height), (0, 0, 0), -1)

        # 1. Sol Kısım: Kasa (POS) Durumu
        if rule_engine.kasa_acik:
            durum_text, durum_color = "KASA: ACIK (RISK)", self.colors["KRITIK"]
        elif rule_engine.cooldown_remaining() > 0:
            durum_text = f"KASA: KAPALI (COOLDOWN {rule_engine.cooldown_remaining():.1f}s)"
            durum_color = self.colors["YUKSEK"]
        else:
            durum_text, durum_color = "KASA: KAPALI (GUVENLI)", self.colors["BILGI"]

        cv2.putText(frame, durum_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, durum_color, 2)

        # 2. Orta Kısım: ALARM uyarısı ya da saat
        if show_alarm:
            cv2.putText(frame, "ALARM: HIRSIZLIK SUPHESI - EL KAYBOLDU!",
                        (width // 2 - 210, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["KRITIK"], 2)
        else:
            current_time = time.strftime("%H:%M:%S")
            cv2.putText(frame, current_time, (width // 2 - 40, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 3. Sağ Kısım: Tuş kısayolları
        cv2.putText(frame, "[o]=Kasa Ac [c]=Kasa Kapat [q]=Cikis", (width - 300, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return frame
