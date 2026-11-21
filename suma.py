import sys
import numpy as np, cv2
print("python:", sys.version.splitlines()[0])
print("exe:", sys.executable)
print("numpy:", np.__version__)
print("opencv:", cv2.__version__)
try:
    import ultralytics
    print("ultralytics:", ultralytics.__version__)
except Exception as e:
    print("ultralytics import failed:", e)
PY