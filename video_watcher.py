import os
import time
import shutil
import cv2
from utils.logger import logger
from core.detector import Detector
from core.rules import TheftRuleEngine
from services.alarm import AlarmManager
from services.database import Database
from ui.overlay import Overlay

INPUT_DIR = "data/input_videos"
PROCESSED_DIR = "data/processed"
DESTEKLENEN_UZANTILAR = (".mp4", ".avi", ".mov", ".mkv")
ALARM_DISPLAY_SECONDS = 2.0

def klasorleri_hazirla():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs("data/snapshots", exist_ok=True)

def videoyu_isle(video_path, detector, rule_engine, alarm_manager, db, overlay):
    video_adi = os.path.basename(video_path)
    camera_name = f"Video-{video_adi}"

    logger.info(f"[W-01] Video analizi basliyor: {video_adi}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"[W-02] Video acilamadi: {video_adi}")
        return

    last_alarm_time = 0.0

    while True:
        success, frame = cap.read()
        if not success:
            break

        hand_detections = detector.detect_and_track(frame)
        alarms = rule_engine.evaluate(hand_detections)
        
        if alarms:
            last_alarm_time = time.time()
        show_alarm = (time.time() - last_alarm_time) < ALARM_DISPLAY_SECONDS

        frame = overlay.draw_zones(frame)
        frame = overlay.draw_detections(frame, hand_detections)
        frame = overlay.draw_status_bar(frame, rule_engine, show_alarm)

        for alarm in alarms:
            alarm_key = f"taraf_{alarm['taraf']}"
            if alarm_manager.should_trigger(alarm_key):
                mesafe_orani = alarm["mesafe_orani"]

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                snapshot_path = os.path.join("data/snapshots", f"{timestamp}_ihlal_{alarm['taraf']}.jpg")
                cv2.imwrite(snapshot_path, frame)

                db.log_event(camera_name, "bolge_ihlali", "KRITIK", mesafe_orani, snapshot_path, alarm["taraf"])

        cv2.imshow(f"Analiz: {video_adi}", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("Kullanici analizi iptal etti.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"[W-03] Analiz tamamlandi: {video_adi}")

def videoyu_tasI(video_path):
    hedef = os.path.join(PROCESSED_DIR, os.path.basename(video_path))
    shutil.move(video_path, hedef)

def main():
    klasorleri_hazirla()

    logger.info("="*50)
    logger.info("  HOT FOLDER İZLEME SİSTEMİ (Zone+Track)")
    logger.info("="*50)

    detector = Detector()
    alarm_manager = AlarmManager()
    db = Database()
    overlay = Overlay()

    islenmis = set()

    while True:
        try:
            dosyalar = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(DESTEKLENEN_UZANTILAR) and f not in islenmis]

            if dosyalar:
                for dosya in dosyalar:
                    video_path = os.path.join(INPUT_DIR, dosya)
                    time.sleep(1)
                    rule_engine = TheftRuleEngine() # Taze motor
                    videoyu_isle(video_path, detector, rule_engine, alarm_manager, db, overlay)
                    videoyu_tasI(video_path)
                    islenmis.add(dosya)
            else:
                time.sleep(2)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"[W-06] Hata: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()
