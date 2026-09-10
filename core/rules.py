import math
import time
from config import settings
from utils.logger import logger

WRIST_HIP_PAIRS = [("sol_bilek", "sol_kalca"), ("sag_bilek", "sag_kalca")]


class TheftRuleEngine:
    """
    Sanal POS logu + iskelet tabanli hirsizlik supheciligi kural motoru.

    Kasanin durumu klavyeden simule edilir (main.py'de 'o'/'c' tuslari).
    Risk penceresi (kasa acik VEYA cooldown suruyorken) aktifken, her el
    (sol/sag) icin:

        1. Bilek-kalca mesafesi (omuz genisligine oranli) COK YAKIN mi?
        2. O bilegin cevresinde el dedektoru BIR EL BULABILDI MI?

    (1) evet VE (2) hayir ise -> ALARM. Elin acik/kapali olmasi, icinde
    para olup olmamasi ONEMLI DEGIL - onemli olan, kasa aciktkan, bilek
    cebe/bele bu kadar yakinken elin görüş alanından fiziksel olarak
    kaybolmus olmasidir.
    """

    def __init__(self):
        self.cooldown_seconds = settings["rules"]["cooldown_seconds"]
        self.proximity_ratio = settings["pose"]["proximity_ratio_threshold"]
        self.match_radius_ratio = settings["pose"]["hand_match_radius_ratio"]

        self.kasa_acik = False
        self.close_time = None

    def kasa_ac(self):
        self.kasa_acik = True
        self.close_time = None
        logger.info("[POS] Kasa ACILDI (simule).")

    def kasa_kapat(self):
        self.kasa_acik = False
        self.close_time = time.time()
        logger.info(f"[POS] Kasa KAPANDI (simule). {self.cooldown_seconds:.0f}s cooldown basladi.")

    def is_risk_window_active(self):
        if self.kasa_acik:
            return True
        if self.close_time is not None and (time.time() - self.close_time) < self.cooldown_seconds:
            return True
        return False

    def cooldown_remaining(self):
        if self.close_time is None:
            return 0.0
        return max(0.0, self.cooldown_seconds - (time.time() - self.close_time))

    @staticmethod
    def _distance(a, b):
        return math.hypot(a["x"] - b["x"], a["y"] - b["y"])

    def _el_bulundu_mu(self, wrist, hand_detections, margin):
        """
        Bilek noktasi, bir el kutusunun (kenardan 'margin' kadar genisletilmis)
        icinde mi diye bakar. Kutunun MERKEZINE mesafe yerine bunu kullaniyoruz
        cunku bilek, elin bir kenarindadir (parmak uclarina dogru uzanan kutunun
        ortasinda degil) - merkez-mesafe olcumu gercek elleri bile kacirabiliyordu.
        """
        for det in hand_detections:
            x1, y1, x2, y2 = det["bbox"]
            if (x1 - margin) <= wrist["x"] <= (x2 + margin) and (y1 - margin) <= wrist["y"] <= (y2 + margin):
                return True
        return False

    def evaluate(self, people, hand_detections):
        """
        people: PoseAnalyzer.analyze() ciktisi (omuz/dirsek/bilek/kalca noktalari)
        hand_detections: Detector.detect() ciktisi ([{"class_name","confidence","bbox"}])

        Return: [{"taraf": "sol_bilek"|"sag_bilek", "mesafe_orani": float}]
        """
        alarms = []

        if not self.is_risk_window_active():
            return alarms

        for person in people:
            omuz_genisligi = person["omuz_genisligi"]
            if omuz_genisligi <= 0:
                continue

            radius = omuz_genisligi * self.match_radius_ratio

            for wrist_name, hip_name in WRIST_HIP_PAIRS:
                wrist = person[wrist_name]
                hip = person[hip_name]

                mesafe_orani = self._distance(wrist, hip) / omuz_genisligi
                if mesafe_orani > self.proximity_ratio:
                    continue  # bilek kalcaya yeterince yakin degil

                if self._el_bulundu_mu(wrist, hand_detections, radius):
                    continue  # el hala gorunur durumda, supheli degil

                logger.warning(
                    f"[R-01] ALARM: {wrist_name} kalcaya asiri yakin "
                    f"(oran {mesafe_orani:.2f}) ve el tespit edilemedi."
                )
                alarms.append({"taraf": wrist_name, "mesafe_orani": mesafe_orani})

        return alarms
