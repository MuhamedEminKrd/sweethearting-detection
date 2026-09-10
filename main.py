import cv2
import sys
import os
import time
from config import settings
from utils.logger import logger
from core.detector import Detector
from core.pose_analyzer import PoseAnalyzer
from core.rules import TheftRuleEngine
from services.alarm import AlarmManager
from services.database import Database
from ui.overlay import Overlay

ALARM_DISPLAY_SECONDS = 2.0  # Ekranda kirmizi ALARM yazisinin kac saniye kalacagi

def main():
    logger.info("[S-03] Sistem başlatılıyor...")

    # 1. Kaynak Ayarı (Kamera mı, Video mu?)
    source_type = settings["source"]["type"]

    if source_type == "camera":
        camera_idx = settings["source"]["camera_index"]
        cap = cv2.VideoCapture(camera_idx)
        camera_name = f"Kamera-{camera_idx}"
        logger.info(f"{camera_name} açılmaya çalışılıyor...")
    else:
        video_path = settings["source"]["video_path"]
        cap = cv2.VideoCapture(video_path)
        camera_name = f"Video-{os.path.basename(video_path)}"
        logger.info(f"{camera_name} açılmaya çalışılıyor...")

    # C-01 Senaryosu: Kaynak bulunamadıysa çök
    if not cap.isOpened():
        logger.critical(f"[C-01] Kaynak açılamadı! (Tipi: {source_type}). Sistem kapatılıyor.")
        sys.exit(1)

    # 2. Sınıflarımızı (Modülleri) Yükleyelim
    try:
        detector = Detector()
        pose_analyzer = PoseAnalyzer()
        rule_engine = TheftRuleEngine()
        alarm_manager = AlarmManager()
        db = Database()
        overlay = Overlay()
    except Exception as e:
        logger.critical(f"Modüller yüklenirken hata: {e}")
        sys.exit(1)

    logger.info("Sistem hazır! ('o'=Kasa Aç, 'c'=Kasa Kapat, 'q'=Çıkış)")

    last_alarm_time = 0.0

    # 3. Ana Döngü
    while True:
        success, frame = cap.read()

        # C-02 Senaryosu: Görüntü kesildiyse veya video bittiyse
        if not success:
            logger.error("[C-02] Görüntü okunamadı veya video bitti. Döngü kırılıyor.")
            break

        # A: Çıkarım - iskelet + tüm kare üzerinde el tespiti
        people = pose_analyzer.analyze(frame)
        hand_detections = detector.detect(frame)

        # B: Kural Motoru (bilek-kalça yakınlığı + el kayboldu mu + kasa açık mı)
        alarms = rule_engine.evaluate(people, hand_detections)
        alarm_taraflari = {a["taraf"] for a in alarms}

        if alarms:
            last_alarm_time = time.time()
        show_alarm = (time.time() - last_alarm_time) < ALARM_DISPLAY_SECONDS

        # C: Çizim
        frame = overlay.draw_pose(frame, people, alarm_taraflari)
        frame = overlay.draw_detections(frame, hand_detections)
        frame = overlay.draw_status_bar(frame, rule_engine, show_alarm)

        # D: Alarm Yönetimi ve Veritabanı Kaydı
        for alarm in alarms:
            alarm_key = f"taraf_{alarm['taraf']}"
            if alarm_manager.should_trigger(alarm_key):
                mesafe_orani = alarm["mesafe_orani"]
                logger.warning(
                    f"🚨 ALARM TETİKLENDİ: {alarm['taraf']} kalçaya aşırı yakın "
                    f"(oran {mesafe_orani:.2f}), el tespit edilemedi."
                )

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                snapshot_dir = "data/snapshots"
                if not os.path.exists(snapshot_dir):
                    os.makedirs(snapshot_dir)
                snapshot_filename = f"{timestamp}_hirsizlik_supheli_{alarm['taraf']}.jpg"
                snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
                cv2.imwrite(snapshot_path, frame)

                db.log_event(camera_name, "el_kayboldu", "KRITIK", mesafe_orani, snapshot_path, alarm["taraf"])

        # E: Ekranda Göster
        cv2.imshow("Kasiyer Kayıp Önleme Sistemi", frame)

        # F: Klavye Kontrolleri
        key = cv2.waitKey(1) & 0xFF
        if key == ord('o'):
            rule_engine.kasa_ac()
        elif key == ord('c'):
            rule_engine.kasa_kapat()
        elif key == ord('q'):
            logger.info("[S-02] Kullanıcı 'q' tuşuna bastı, çıkış yapılıyor...")
            break

    # Döngü bittiğinde temizlik yap (S-04 Senaryosu)
    cap.release()
    cv2.destroyAllWindows()
    logger.info("[S-04] Kaynak serbest bırakıldı, pencereler kapatıldı. Sistem durduruldu.")

if __name__ == "__main__":
    main()
