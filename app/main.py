# app/main.py (improved, robust PPE monitor)
import os
import sys
import time
import traceback
from pathlib import Path
import logging
from dotenv import load_dotenv

# ensure project root is on sys.path when executed directly (safety)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# now safe to import local modules
try:
    from utils.init_db import insert_violation, init_db
except Exception as e:
    # Provide a harmless fallback if utils not available — app can still run
    def init_db():
        logging.warning("init_db() not available: %s", e)

    def insert_violation(*args, **kwargs):
        logging.warning("insert_violation() not available: %s", e)

# lazy import of heavy libs after env loaded
load_dotenv()

# Configuration (can be overridden in .env)
MODEL_PATH = os.getenv("MODEL_PATH", "runs/detect/train4/weights/best.pt")
SOURCE = os.getenv("SOURCE", "0")              # "0" for webcam or file/rtsp path
VIOLATION_CONF = float(os.getenv("VIOLATION_CONF", "0.5"))
NO_PPE_PREFIX = os.getenv("NO_PPE_PREFIX", "no_")
SAVE_DIR = Path(os.getenv("SAVE_DIR", "violations"))
SAVE_DIR.mkdir(parents=True, exist_ok=True)
VIOLATION_COOLDOWN = float(os.getenv("VIOLATION_COOLDOWN", "5.0"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "1024"))
DEBUG = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

# configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# attempt to choose device
DEVICE = "cpu"
try:
    import torch

    if os.getenv("USE_CUDA", "auto").lower() in ("1", "true", "yes") and torch.cuda.is_available():
        DEVICE = "cuda"
except Exception:
    DEVICE = "cpu"

# delayed heavy imports
try:
    import cv2
except Exception as e:
    logging.critical("OpenCV is required (cv2). Install opencv-python into your active venv. Error: %s", e)
    raise

try:
    from ultralytics import YOLO
except Exception as e:
    logging.critical("Ultralytics (YOLO) import failed. Install ultralytics in your venv. Error: %s", e)
    raise


def to_int_if_digit(s):
    """Convert camera index string to int if possible, otherwise return string path."""
    try:
        return int(s)
    except Exception:
        return s


def normalize_label(label: str) -> str:
    """Normalize label for robust matching: lowercase, underscores, strip whitespace."""
    return label.strip().lower().replace(" ", "_")


def draw_box(frame, x1, y1, x2, y2, label, conf, color=(0, 255, 0)):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    cv2.putText(frame, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def load_model(path: str):
    p = Path(path)
    if not p.exists():
        logging.warning("Model not found at %s. Please update MODEL_PATH in .env", path)
    logging.info("Loading model %s on device %s", path, DEVICE)
    model = YOLO(path)
    # attempt to move to chosen device if supported
    try:
        model.to(DEVICE)
    except Exception:
        # some ultralytics builds don't support .to() — ignore
        pass
    # ensure names are normalized mapping
    names = model.names
    # convert to dict {id: normalized_name}
    if isinstance(names, dict):
        norm_names = {int(k): normalize_label(v) for k, v in names.items()}
    else:
        norm_names = {i: normalize_label(v) for i, v in enumerate(names)}
    model._norm_names = norm_names
    logging.debug("Model class mapping: %s", norm_names)
    return model


def should_mark_violation(label_norm: str) -> bool:
    """Return True if this normalized label indicates missing PPE."""
    # consider 'no', 'no_*' or labels that contain 'no' token
    if label_norm.startswith(NO_PPE_PREFIX) or label_norm.startswith("no_"):
        return True
    # catch variants: "no", "no_gloves", "no-gloves", "nogloves", "no gloves"
    if label_norm == "no" or label_norm.startswith("no"):
        return True
    # safer fallback: token 'no' in parts
    if "no" in label_norm.split("_"):
        return True
    return False


def main():
    init_db()  # best-effort (won't crash if missing)

    # load model
    model = load_model(MODEL_PATH)

    # open source
    src = to_int_if_digit(SOURCE)
    logging.info("Opening video source: %s", src)
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if os.name == "nt" else 0)
    # set camera resolution to help small-object detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        logging.critical("Cannot open video source: %s", SOURCE)
        raise RuntimeError("Cannot open video source: " + str(SOURCE))

    last_saved_time = 0.0
    logging.info("Started video capture. Press 'q' in the window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning("Empty frame received, retrying...")
                time.sleep(0.05)
                continue

            # run inference (use imgsz and a lower conf in DEBUG)
            conf_used = 0.02 if DEBUG else VIOLATION_CONF
            try:
                results = model(frame, imgsz=int(IMG_SIZE), conf=conf_used, verbose=False)
            except Exception as e:
                logging.error("Model inference failed: %s", e)
                results = []

            if not results:
                # show frame and continue
                cv2.imshow("PPE Monitor (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            r = results[0]
            boxes = getattr(r, "boxes", None)
            violation_detected = False
            missing_labels = []
            top_conf = 0.0

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    try:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        # normalized label
                        label_norm = model._norm_names.get(cls_id, str(cls_id))
                        # double-safety: if names missing, fallback to raw
                        label_raw = model.names[cls_id] if hasattr(model, "names") else label_norm
                        # skip extremely low conf (already filtered by conf_used)
                        if conf < (VIOLATION_CONF * 0.3) and not DEBUG:
                            continue

                        xyxy = None
                        try:
                            # handle both tensor and numpy outputs
                            xy = box.xyxy[0]
                            if hasattr(xy, "cpu"):
                                xyxy = xy.cpu().numpy()
                            else:
                                xyxy = xy.numpy()
                        except Exception:
                            # last-resort: try to use r.boxes.xyxy if available
                            try:
                                xyxy = r.boxes.xyxy[0].cpu().numpy()
                            except Exception:
                                logging.debug("Could not read coordinates for box")
                                continue

                        x1, y1, x2, y2 = map(int, xyxy)
                        color = (0, 255, 0)
                        if should_mark_violation(label_norm):
                            color = (0, 0, 255)
                            violation_detected = True
                            missing_labels.append(label_norm)
                        top_conf = max(top_conf, conf)
                        # draw label (use raw for readability if available)
                        draw_box(frame, x1, y1, x2, y2, label_raw, conf, color=color)
                        if DEBUG:
                            logging.debug("DETECTED: %s cls=%d conf=%.3f xy=%s", label_raw, cls_id, conf, (x1, y1, x2, y2))
                    except Exception as e:
                        logging.debug("Box loop error: %s", e)
                        continue

            now = time.time()
            if violation_detected and (now - last_saved_time) >= VIOLATION_COOLDOWN:
                ts = time.strftime("%Y%m%d_%H%M%S")
                fname = SAVE_DIR / f"violation_{ts}.jpg"
                try:
                    cv2.imwrite(str(fname), frame)
                    insert_violation(str(fname), "camera_1", ",".join(missing_labels) or "unknown", float(top_conf))
                    logging.info("Violation saved: %s missing=%s conf=%.2f", fname, missing_labels, top_conf)
                except Exception as e:
                    logging.error("Failed saving/inserting violation: %s", e)
                last_saved_time = now

            # show frame
            cv2.imshow("PPE Monitor (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logging.info("User requested exit.")
                break

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception:
        logging.critical("Fatal error in capture loop:")
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


