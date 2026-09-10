import sqlite3
import os
from datetime import datetime
from config import settings
from utils.logger import logger

class Database:
    def __init__(self):
        self.db_path = settings["database"]["path"]
        self._init_db()

    def _init_db(self):
        """Veritabanı ve tabloyu oluşturur (Eğer yoksa)"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS olaylar (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tarih TEXT,
                    kamera_id TEXT,
                    tehlike_tipi TEXT,
                    seviye TEXT,
                    fotograf_yolu TEXT
                )
            ''')

            # Sütunlar daha sonradan eklendiği için yoksa ekliyoruz (Migration)
            cursor.execute("PRAGMA table_info(olaylar)")
            columns = [info[1] for info in cursor.fetchall()]
            if 'guven_orani' not in columns:
                cursor.execute("ALTER TABLE olaylar ADD COLUMN guven_orani REAL")
            if 'taraf' not in columns:
                cursor.execute("ALTER TABLE olaylar ADD COLUMN taraf TEXT")

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[DB-01] Veritabanı başlatılamadı: {e}")

    def log_event(self, camera_id, danger_type, severity, mesafe_orani, snapshot_path, taraf=None):
        """Tehlike anını veritabanına kaydeder"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('''
                INSERT INTO olaylar (tarih, kamera_id, tehlike_tipi, seviye, guven_orani, fotograf_yolu, taraf)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (now, camera_id, danger_type, severity, mesafe_orani, snapshot_path, taraf))
            conn.commit()
            conn.close()
            logger.info(f"[DB-02] Veritabanına kaydedildi: [{camera_id}] {danger_type} (oran {mesafe_orani:.2f}) - Taraf: {taraf} - Fotoğraf: {snapshot_path}")
        except Exception as e:
            logger.error(f"[DB-03] Veritabanı kayıt hatası: {e}")
