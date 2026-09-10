import time
import cv2
import numpy as np
from config import settings
from utils.logger import logger

class TheftRuleEngine:
    """
    Bolge (ROI) ve Takip (Tracking) tabanli yeni hirsizlik kural motoru.
    İskelet (pose) yerine ellerin bolgeler arasi gecisini takip eder.
    """

    def __init__(self):
        # Ayarlari oku
        self.zones = settings.get("zones", {})
        
        # Poligonlari numpy dizilerine cevir
        self.polygons = {}
        for zone_name in ["etkilesim", "kasa", "tehlike"]:
            points = self.zones.get(zone_name, [])
            if len(points) >= 3:
                self.polygons[zone_name] = np.array(points, np.int32)
            else:
                self.polygons[zone_name] = None
                if zone_name != "etkilesim": # Etkilesim sart degil ama kasa ve tehlike sart
                    logger.warning(f"Bolge eksik: {zone_name}. setup_zones.py ile cizmeniz onerilir.")

        # track_id bazli gecmis: { track_id: {"visited_etkilesim": bool, "visited_kasa": bool, "alarm_triggered": bool, "last_seen": float} }
        self.track_history = {}
        
        # Artik klavyeden kasa acma simülasyonu yok ama overlay.py patlamasin diye mock ozellikler
        self.kasa_acik = False

    def kasa_ac(self):
        pass # Artik kullanilmiyor

    def kasa_kapat(self):
        pass # Artik kullanilmiyor
        
    def cooldown_remaining(self):
        return 0.0 # Artik kullanilmiyor

    def _get_zone(self, center_x, center_y):
        """Merkez noktasinin hangi poligona dustugunu bulur."""
        pt = (float(center_x), float(center_y))
        
        # Oncelik sirasi: Tehlike > Kasa > Etkilesim
        if self.polygons.get("tehlike") is not None:
            if cv2.pointPolygonTest(self.polygons["tehlike"], pt, False) >= 0:
                return "tehlike"
                
        if self.polygons.get("kasa") is not None:
            if cv2.pointPolygonTest(self.polygons["kasa"], pt, False) >= 0:
                return "kasa"
                
        if self.polygons.get("etkilesim") is not None:
            if cv2.pointPolygonTest(self.polygons["etkilesim"], pt, False) >= 0:
                return "etkilesim"
                
        return "disari"

    def evaluate(self, hand_detections):
        """
        hand_detections: Detector.detect_and_track() ciktisi
        Return: [{"taraf": "El-ID", "mesafe_orani": 1.0 (Mock)}]
        """
        alarms = []
        current_time = time.time()
        
        # aktif track_id'leri topla
        active_track_ids = set()

        for det in hand_detections:
            track_id = det["track_id"]
            if track_id == -1:
                continue # Henuz ID atanmamis
                
            active_track_ids.add(track_id)
            
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            
            # Gecmis kaydi yoksa olustur
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    "time_etkilesim": 0.0, 
                    "time_kasa": 0.0, 
                    "alarm_triggered": False,
                    "last_seen": current_time,
                    "last_zone": "disari"
                }
                
            history = self.track_history[track_id]
            history["last_seen"] = current_time
            
            current_zone = self._get_zone(cx, cy)
            
            # Guncel bolgeyi kaydet
            if current_zone != "disari":
                history["last_zone"] = current_zone
            
            if current_zone == "etkilesim":
                history["time_etkilesim"] = current_time
            elif current_zone == "kasa":
                history["time_kasa"] = current_time

        # Kaybolan elleri kontrol et (0.9 saniye sarti)
        for t_id, history in list(self.track_history.items()):
            time_since_last_seen = current_time - history["last_seen"]
            
            # Ziyaretlerin ustunden cok zaman gectiyse (ornek: 15 saniye), bunlari "eski" say.
            # Yani kasiyer 15 saniye once kasaya dokunup simdi elini cebine atiyorsa bu hirsizlik degildir.
            visited_etkilesim_recently = (current_time - history["time_etkilesim"]) < 15.0 and history["time_etkilesim"] > 0
            visited_kasa_recently = (current_time - history["time_kasa"]) < 15.0 and history["time_kasa"] > 0
            
            # Eger el 0.9 saniyeden fazladir kayipsa ve en son Tehlike bolgesindeyse
            if time_since_last_seen >= 0.9 and not history["alarm_triggered"]:
                if history["last_zone"] == "tehlike":
                    # SENARYO 1: Musteriden alip cebe atma (Etkilesim -> Tehlike -> Kaybolma)
                    if visited_etkilesim_recently and not visited_kasa_recently:
                        logger.warning(f"[R-01] ALARM: El-{t_id} kasaya ugramadan cebe gitti ve KAYBOLDU (>{time_since_last_seen:.1f}s)!")
                        alarms.append({"taraf": f"El-{t_id}", "mesafe_orani": 1.0})
                        history["alarm_triggered"] = True
                        
                    # SENARYO 2: Kasadan para alip cebe atma (Kasa -> Tehlike -> Kaybolma)
                    elif visited_kasa_recently:
                        logger.warning(f"[R-02] ALARM: El-{t_id} kasadan cikip cebe/bele gitti ve KAYBOLDU (>{time_since_last_seen:.1f}s)!")
                        alarms.append({"taraf": f"El-{t_id}", "mesafe_orani": 1.0})
                        history["alarm_triggered"] = True

            # Uzun sure gorunmeyen (orn: 5 saniye) track_id'leri temizle (Memory leak onleme)
            if time_since_last_seen > 5.0:
                del self.track_history[t_id]

        return alarms
