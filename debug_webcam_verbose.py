import cv2, os
from ultralytics import YOLO

MODEL = "runs/detect/train4/weights/best.pt"
CONF = 0.03
IMGSZ = 1024

print("Loading model:", MODEL)
m = YOLO(MODEL)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret, frame = cap.read()
cap.release()
if not ret:
    print("ERROR: could not read frame"); raise SystemExit

print("Frame shape:", frame.shape)
res = m(frame, conf=CONF, imgsz=IMGSZ, verbose=False)
r = res[0]
boxes = getattr(r, "boxes", None)
print("Boxes:", 0 if boxes is None else len(boxes))
if boxes:
    for i, box in enumerate(boxes):
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        name = m.names[cls] if hasattr(m, "names") else str(cls)
        print(f"Box {i}: cls={cls} name={name} conf={conf:.4f} xyxy={xyxy}")

os.makedirs("runs/debug", exist_ok=True)
try:
    img = r.plot()
    if img is not None:
        import cv2
        cv2.imwrite("runs/debug/webcam_debug.jpg", img[:,:,::-1])
        print("Saved annotated debug image to runs/debug/webcam_debug.jpg")
    else:
        cv2.imwrite("runs/debug/webcam_debug.jpg", frame)
        print("Saved raw frame to runs/debug/webcam_debug.jpg (plot returned None)")
except Exception as e:
    print("Save error:", e)
    cv2.imwrite("runs/debug/webcam_debug.jpg", frame)
    print("Saved raw frame to runs/debug/webcam_debug.jpg")
