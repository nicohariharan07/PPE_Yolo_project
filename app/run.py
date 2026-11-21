from ultralytics import YOLO
model = YOLO("runs/detect/train6/weights/best.pt")
model.predict(source=0, show=True)
