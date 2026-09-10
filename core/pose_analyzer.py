from ultralytics import YOLO
from config import settings
from utils.logger import logger

# COCO 17-keypoint index sirasi (yolo26/yolov8-pose ile ayni)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


class PoseAnalyzer:
    """
    Vucut iskeletini (omuz/dirsek/bilek/kalca) cikarir. Hazir/onceden egitilmis
    yolo26x-pose modelini kullanir - ozel egitim gerektirmez.
    """

    def __init__(self):
        self.model_path = settings["model"]["pose_path"]
        logger.info(f"Pose modeli yukleniyor: {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            logger.critical(f"[M-04] Pose modeli yuklenemedi: {e}")
            raise e

    @staticmethod
    def _point(kpts, idx):
        x, y, v = kpts[idx].tolist()
        return {"x": x, "y": y, "v": v}

    def analyze(self, frame):
        """
        Karedeki her kisi icin omuz/dirsek/bilek/kalca noktalarini ve
        omuz genisligini (normalizasyon referansi) dondurur.

        Return: [{"sol_omuz","sag_omuz","sol_dirsek","sag_dirsek",
                  "sol_bilek","sag_bilek","sol_kalca","sag_kalca",
                  "omuz_genisligi"}, ...]
        """
        results = self.model.predict(source=frame, verbose=False)

        people = []
        if len(results) == 0 or results[0].keypoints is None:
            return people

        for kpts in results[0].keypoints.data:
            sol_omuz = self._point(kpts, L_SHOULDER)
            sag_omuz = self._point(kpts, R_SHOULDER)
            omuz_genisligi = ((sol_omuz["x"] - sag_omuz["x"]) ** 2 + (sol_omuz["y"] - sag_omuz["y"]) ** 2) ** 0.5

            people.append({
                "sol_omuz": sol_omuz,
                "sag_omuz": sag_omuz,
                "sol_dirsek": self._point(kpts, L_ELBOW),
                "sag_dirsek": self._point(kpts, R_ELBOW),
                "sol_bilek": self._point(kpts, L_WRIST),
                "sag_bilek": self._point(kpts, R_WRIST),
                "sol_kalca": self._point(kpts, L_HIP),
                "sag_kalca": self._point(kpts, R_HIP),
                "omuz_genisligi": omuz_genisligi,
            })

        return people
