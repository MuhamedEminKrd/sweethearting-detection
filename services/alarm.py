import time
from config import settings
from utils.logger import logger

class AlarmManager:
    def __init__(self):
        self.debounce_seconds = settings["rules"]["debounce_seconds"]
        self.last_alarm_times = {}

    def should_trigger(self, object_name):
        """
        Eğer nesne için belirlenen bekleme süresi (debounce) geçtiyse True döner.
        Geçmediyse (spam yapıyorsa) False döner.
        """
        if not object_name:
            return False
            
        current_time = time.time()
        last_time = self.last_alarm_times.get(object_name, 0)

        if current_time - last_time >= self.debounce_seconds:
            self.last_alarm_times[object_name] = current_time
            return True
            
        return False
