# pyrefly: ignore [missing-import]
from ultralytics import YOLO
# pyrefly: ignore [missing-import]
import torch
import os
from utils.logger import logger

# Ultralytics, 'project' GORELI bir yol verilirse onu global settings.json'daki
# 'runs_dir' (kullanici genelinde, tum projeler icin ortak bir ayar) ile
# birlestiriyor - bu yuzden ciktimiz kendi proje klasorumuze degil
# C:\Users\<kullanici>\runs\ altina gidiyordu. Mutlak yol vererek bu genel
# ayari devre disi birakiyoruz, cikti kesin olarak bu proje klasorunun
# icinde kalir.
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

def main():
    if not torch.cuda.is_available():
        logger.critical("CUDA bulunamadı! Eğitim iptal edildi.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"GPU Algılandı: {gpu_name}. Eğitim başlıyor!")

    model = YOLO("yolov8s.pt")

    try:
        # NOT: datasets/ eski bıçak/silah/ateş veri seti archive/datasets/'e taşındı.
        # Yeni para_var/para_yok veri seti (Roboflow'dan indirilecek) buraya, datasets/ altına konmalı.
        results = model.train(
            data="datasets/data.yaml",
            epochs=50,
            imgsz=640,
            batch=8,          # 16'da AMP kontrolü sirasinda sessizce cakiliyordu (6GB VRAM yetersiz), 8'e dusuruldu.
            device=0,         # RTX 4050
            amp=False,        # AMP acikken 2. epoch'tan itibaren cls_loss patliyordu (NaN) - kapatildi
            lr0=0.001,        # Varsayilan 0.01, bu kadar kucuk veri setinde (88 gorsel) patlamaya sebep oluyordu
            cache=True,       # Sadece ~125 gorsel var, RAM'e sigar, disk I/O sıfıra düşer
            workers=4,        # 4 paralel CPU ile veri hazırlama
            optimizer="AdamW", # YOLOv8 için en stabil optimizer
            project=RUNS_DIR, # Mutlak yol - global runs_dir ayarini devre disi birakir
            name="el_modelim_v1",
            patience=15,      # 15 epoch boyunca iyileşme olmazsa erken durdurur (zaman tasarrufu)
            verbose=True
        )
        logger.info("Eğitim tamamlandı!")
        logger.info(f"En iyi model: {os.path.join(RUNS_DIR, 'detect', 'el_modelim_v1', 'weights', 'best.pt')}")

    except torch.cuda.OutOfMemoryError:
        logger.error("[M-03] VRAM yetersiz! batch değerini 8'e düşür ve tekrar dene.")
    except Exception as e:
        logger.error(f"Eğitim hatası: {e}")

if __name__ == "__main__":
    main()
