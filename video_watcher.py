import os
import time
import shutil
import cv2
from utils.logger import logger
from core.detector import Detector
from core.pose_analyzer import PoseAnalyzer
from core.rules import TheftRuleEngine
from services.alarm import AlarmManager
from services.database import Database
from ui.overlay import Overlay

# İzlenecek ve işlenen videoların klasörleri
INPUT_DIR = "data/input_videos"
PROCESSED_DIR = "data/processed"
DESTEKLENEN_UZANTILAR = (".mp4", ".avi", ".mov", ".mkv")
ALARM_DISPLAY_SECONDS = 2.0

def klasorleri_hazirla():
    """Gerekli klasörler yoksa oluşturur."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs("data/snapshots", exist_ok=True)

def videoyu_isle(video_path, detector, pose_analyzer, rule_engine, alarm_manager, db, overlay):
    """Tek bir videoyu baştan sona analiz eder."""
    video_adi = os.path.basename(video_path)
    camera_name = f"Video-{video_adi}"

    logger.info(f"[W-01] Video analizi başlıyor: {video_adi}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"[W-02] Video açılamadı: {video_adi}")
        return

    last_alarm_time = 0.0

    while True:
        success, frame = cap.read()
        if not success:
            break

        people = pose_analyzer.analyze(frame)
        hand_detections = detector.detect(frame)

        alarms = rule_engine.evaluate(people, hand_detections)
        alarm_taraflari = {a["taraf"] for a in alarms}
        if alarms:
            last_alarm_time = time.time()
        show_alarm = (time.time() - last_alarm_time) < ALARM_DISPLAY_SECONDS

        frame = overlay.draw_pose(frame, people, alarm_taraflari)
        frame = overlay.draw_detections(frame, hand_detections)
        frame = overlay.draw_status_bar(frame, rule_engine, show_alarm)

        for alarm in alarms:
            alarm_key = f"taraf_{alarm['taraf']}"
            if alarm_manager.should_trigger(alarm_key):
                mesafe_orani = alarm["mesafe_orani"]
                logger.warning(
                    f"🚨 ALARM: {alarm['taraf']} kalçaya aşırı yakın (oran {mesafe_orani:.2f}) "
                    f"el tespit edilemedi - {video_adi}"
                )

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                snapshot_filename = f"{timestamp}_hirsizlik_supheli_{alarm['taraf']}.jpg"
                snapshot_path = os.path.join("data/snapshots", snapshot_filename)
                cv2.imwrite(snapshot_path, frame)

                db.log_event(camera_name, "el_kayboldu", "KRITIK", mesafe_orani, snapshot_path, alarm["taraf"])

        # Ekranda göster ('q' ile iptal, 'o'/'c' ile POS simülasyonu test edilebilir)
        cv2.imshow(f"Analiz: {video_adi}", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('o'):
            rule_engine.kasa_ac()
        elif key == ord('c'):
            rule_engine.kasa_kapat()
        elif key == ord('q'):
            logger.info("Kullanıcı analizi iptal etti.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"[W-03] Analiz tamamlandı: {video_adi}")


def videoyu_tasI(video_path):
    """İşlenen videoyu 'processed' klasörüne taşır."""
    hedef = os.path.join(PROCESSED_DIR, os.path.basename(video_path))
    shutil.move(video_path, hedef)
    logger.info(f"[W-04] Video taşındı: {hedef}")


def main():
    klasorleri_hazirla()

    logger.info("="*50)
    logger.info("  HOT FOLDER İZLEME SİSTEMİ BAŞLADI")
    logger.info(f"  Klasör: {os.path.abspath(INPUT_DIR)}")
    logger.info("  Buraya video atın, sistem otomatik analiz edecek!")
    logger.info("  Durdurmak için Ctrl+C basın.")
    logger.info("="*50)

    # Modülleri bir kez yükle, her video için yeniden yükleme
    detector = Detector()
    pose_analyzer = PoseAnalyzer()
    alarm_manager = AlarmManager()
    db = Database()
    overlay = Overlay()

    islenmis = set()  # Aynı videoyu iki kez işlememek için

    while True:
        try:
            dosyalar = [
                f for f in os.listdir(INPUT_DIR)
                if f.lower().endswith(DESTEKLENEN_UZANTILAR) and f not in islenmis
            ]

            if dosyalar:
                for dosya in dosyalar:
                    video_path = os.path.join(INPUT_DIR, dosya)

                    # Dosyanın tamamen kopyalanmış olduğunu bekle (kısmen kopyalı olabilir)
                    time.sleep(1)

                    # Her video için taze bir kural motoru (kasa durumu videolar arası karışmasın)
                    rule_engine = TheftRuleEngine()

                    videoyu_isle(video_path, detector, pose_analyzer, rule_engine, alarm_manager, db, overlay)
                    videoyu_tasI(video_path)
                    islenmis.add(dosya)
            else:
                # Yeni video yok, 2 saniye bekle ve tekrar kontrol et
                time.sleep(2)

        except KeyboardInterrupt:
            logger.info("\n[W-05] Sistem kullanıcı tarafından durduruldu.")
            break
        except Exception as e:
            logger.error(f"[W-06] Beklenmeyen hata: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()
