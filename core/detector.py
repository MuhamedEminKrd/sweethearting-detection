from ultralytics import YOLO
from config import settings
from utils.logger import logger

class Detector:
    """
    Tek sinifli 'el' dedektoru. Elin acik mi kapali mi, icinde para olup
    olmadigi onemli degil - sadece "burada bir el var mi" sorusuna cevap verir.
    """

    def __init__(self):
        self.model_path = settings["model"]["hand_path"]
        self.threshold = settings["model"]["hand_threshold"]

        logger.info(f"El dedektoru yukleniyor: {self.model_path}")

        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Model basariyla yuklendi. Siniflar: {list(self.model.names.values())}")
        except Exception as e:
            logger.critical(f"[M-02] Model yuklenemedi: {e}")
            raise e

    def detect_and_track(self, frame):
        """
        Kameradan gelen tek bir kareyi alir, icindeki elleri ByteTrack ile takip eder.
        Geriye tespitlerin listesini dondurur (track_id dahil).
        """
        results = self.model.track(source=frame, conf=self.threshold, persist=True, tracker="bytetrack.yaml", verbose=False)

        detections = []

        if len(results) == 0 or len(results[0].boxes) == 0:
            return detections

        boxes = results[0].boxes

        for box in boxes:
            class_id = int(box.cls[0].item())
            class_name = self.model.names[class_id]
            confidence = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Takip ediliyorsa ID alinir, henuz ID atanmadiysa None doner
            track_id = int(box.id[0].item()) if box.id is not None else -1

            detections.append({
                "class_name": class_name,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
                "track_id": track_id
            })

        return detections
