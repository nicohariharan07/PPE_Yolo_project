# Example wrapper to start YOLOv10 training via ultralytics
from ultralytics import YOLO
import os

def train():
    model_cfg = "yolov10n.yaml"  # or path to your model config
    data_cfg = "data/ppe.yaml"
    epochs = int(os.getenv("EPOCHS", "100"))
    imgsz = int(os.getenv("IMGSZ", "640"))
    batch = int(os.getenv("BATCH", "8"))

    model = YOLO(model_cfg)
    model.train(data=data_cfg, epochs=epochs, imgsz=imgsz, batch=batch)

if __name__ == "__main__":
    train()
