import cv2
import sys
import os
import time
from config import settings
from utils.logger import logger
from core.detector import Detector
from core.rules import TheftRuleEngine
from services.alarm import AlarmManager
from services.database import Database
from ui.overlay import Overlay

ALARM_DISPLAY_SECONDS = 2.0

def main():
    logger.info("[S-03] Sistem baslatiliyor (Bolge + Takip Mimarisi)...")

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        cap = cv2.VideoCapture(video_path)
        camera_name = f"Video-{os.path.basename(video_path)}"
        logger.info(f"[S-01] Komut satirindan video acildi: {video_path}")
    else:
        source_type = settings["source"]["type"]
        if source_type == "camera":
            camera_idx = settings["source"]["camera_index"]
            cap = cv2.VideoCapture(camera_idx)
            camera_name = f"Kamera-{camera_idx}"
        else:
            video_path = settings["source"]["video_path"]
            cap = cv2.VideoCapture(video_path)
            camera_name = f"Video-{os.path.basename(video_path)}"

    if not cap.isOpened():
        logger.critical(f"[C-01] Kaynak acilamadi! Sistem kapatiliyor.")
        sys.exit(1)

    try:
        detector = Detector()
        rule_engine = TheftRuleEngine()
        alarm_manager = AlarmManager()
        db = Database()
        overlay = Overlay()
    except Exception as e:
        logger.critical(f"Moduller yuklenirken hata: {e}")
        sys.exit(1)

    logger.info("Sistem hazir! Cikis icin 'q' tusuna basin.")

    last_alarm_time = 0.0

    while True:
        success, frame = cap.read()
        if not success:
            logger.error("[C-02] Goruntu okunamadi veya bitti.")
            break

        # A: Cikarim - ByteTrack ile el takibi
        hand_detections = detector.detect_and_track(frame)

        # B: Kural Motoru (Bolge Gecisi)
        alarms = rule_engine.evaluate(hand_detections)

        if alarms:
            last_alarm_time = time.time()
        show_alarm = (time.time() - last_alarm_time) < ALARM_DISPLAY_SECONDS

        # C: Cizim
        frame = overlay.draw_zones(frame)
        frame = overlay.draw_detections(frame, hand_detections)
        frame = overlay.draw_status_bar(frame, rule_engine, show_alarm)

        # D: Alarm Yonetimi ve Kayit
        for alarm in alarms:
            alarm_key = f"taraf_{alarm['taraf']}"
            if alarm_manager.should_trigger(alarm_key):
                mesafe_orani = alarm["mesafe_orani"]

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                snapshot_dir = "data/snapshots"
                if not os.path.exists(snapshot_dir):
                    os.makedirs(snapshot_dir)
                snapshot_filename = f"{timestamp}_ihlal_{alarm['taraf']}.jpg"
                snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
                cv2.imwrite(snapshot_path, frame)

                db.log_event(camera_name, "bolge_ihlali", "KRITIK", mesafe_orani, snapshot_path, alarm["taraf"])

        cv2.imshow("Kasiyer Kayip Onleme Sistemi (Zone+Track)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("[S-02] Kullanici 'q' ile cikis yapti.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("[S-04] Sistem durduruldu.")

if __name__ == "__main__":
    main()
