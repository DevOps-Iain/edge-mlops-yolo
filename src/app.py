# src/app.py
from ultralytics import YOLO
import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from datetime import datetime
from functools import lru_cache
import os
import time
import traceback

app = FastAPI()

# ----------------------------
# Paths (local + docker friendly)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(PROJECT_ROOT, "best.onnx"))
FEEDBACK_DIR = os.getenv("FEEDBACK_DIR", os.path.join(PROJECT_ROOT, "feedback_data"))
os.makedirs(FEEDBACK_DIR, exist_ok=True)

print(f"Feedback directory initialized: {FEEDBACK_DIR}")
print(f"Model path: {MODEL_PATH}")

# ----------------------------
# Camera config
# ----------------------------
def _default_camera_device():
    # In Linux containers, /dev/video0 usually exists; but "by name + CAP_V4L2" may be unstable.
    # We'll still default to /dev/video0, but open with CAP_ANY first.
    if os.path.exists("/dev/video0"):
        return "/dev/video0"
    return "0"  # local dev fallback

CAMERA_DEVICE_RAW = os.getenv("CAMERA_DEVICE", _default_camera_device()).strip()

# Default sizes: match your Pi USB cam 352x288; otherwise 640x480
if os.path.exists("/dev/video0"):
    DEFAULT_W, DEFAULT_H = 352, 288
else:
    DEFAULT_W, DEFAULT_H = 640, 480

CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", str(DEFAULT_W)))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", str(DEFAULT_H)))

def _parse_camera_device(raw: str):
    raw = str(raw).strip()
    if raw.isdigit():
        return int(raw)  # index 0/1/2...
    return raw          # path like /dev/video0

CAMERA_DEVICE = _parse_camera_device(CAMERA_DEVICE_RAW)

# ----------------------------
# Model load (lazy + cached)
# ----------------------------
@lru_cache(maxsize=1)
def get_model():
    return YOLO(MODEL_PATH, task="detect")

def _apply_camera_settings(cap: cv2.VideoCapture):
    # MJPG is often more stable if supported; harmless if not.
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    # Reduce buffering (may be ignored depending on backend)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

def _open_camera():
    """
    Robust open strategy:
    1) Try CAP_ANY with device path (e.g., '/dev/video0') OR index
    2) If failed and device is a path, fallback to index 0 (CAP_ANY)
    3) Only then try CAP_V4L2 with index 0 (some builds dislike "by name" with CAP_V4L2)
    """
    attempts = []

    # Attempt 1: CAP_ANY (recommended)
    attempts.append(("CAP_ANY", lambda: cv2.VideoCapture(CAMERA_DEVICE)))

    # Attempt 2: if a named device path, fallback to index 0 with CAP_ANY
    if isinstance(CAMERA_DEVICE, str) and CAMERA_DEVICE.startswith("/dev/"):
        attempts.append(("CAP_ANY index0", lambda: cv2.VideoCapture(0)))

    # Attempt 3: CAP_V4L2 index0 (only as last resort)
    attempts.append(("CAP_V4L2 index0", lambda: cv2.VideoCapture(0, cv2.CAP_V4L2)))

    for name, fn in attempts:
        cap = fn()
        _apply_camera_settings(cap)
        opened = cap.isOpened()
        print(f"[camera] open attempt={name}, device_raw={CAMERA_DEVICE_RAW}, parsed={CAMERA_DEVICE}, opened={opened}")
        if opened:
            return cap
        cap.release()

    # return a closed cap as signal
    return cv2.VideoCapture()

def _reopen_camera(cap: cv2.VideoCapture, retries: int = 3, sleep_s: float = 0.5):
    try:
        cap.release()
    except Exception:
        pass
    for i in range(retries):
        time.sleep(sleep_s)
        cap = _open_camera()
        if cap.isOpened():
            # warm-up read
            _ = cap.read()
            print(f"[camera] reopen success on attempt {i+1}/{retries}")
            return cap
        print(f"[camera] reopen failed on attempt {i+1}/{retries}")
    return cap

def get_frames():
    cap = _open_camera()
    if not cap.isOpened():
        print("🔴 ERROR: Camera not accessible. Check device mapping/permissions or set CAMERA_DEVICE=0.")
        time.sleep(1)
        return

    # Model: fail-safe (keep raw stream)
    model = None
    try:
        model = get_model()
    except Exception as e:
        print(f"🟡 WARN: model load failed, fallback to raw frames. err={e}")
        print(traceback.format_exc())

    try:
        while True:
            success, frame = cap.read()

            if not success or frame is None:
                print("🔴 ERROR: Failed to read frame. Trying to reopen camera...")
                cap = _reopen_camera(cap, retries=3, sleep_s=0.5)
                if not cap.isOpened():
                    print("🔴 ERROR: Camera reopen failed. Stop stream.")
                    break
                continue

            annotated_frame = frame
            result_for_feedback = None

            if model is not None:
                try:
                    results = model(frame, stream=True, verbose=False)
                    result = next(results, None)
                    if result is not None:
                        result_for_feedback = result
                        annotated_frame = result.plot()
                except Exception as e:
                    print(f"🟡 WARN: inference failed, fallback to raw frame. err={e}")

            # Feedback capture
            if result_for_feedback is not None:
                try:
                    if hasattr(result_for_feedback, "boxes") and result_for_feedback.boxes is not None and len(result_for_feedback.boxes) > 0:
                        min_conf = float(result_for_feedback.boxes.conf.min().item())
                        if 0.25 < min_conf < 0.5:
                            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                            file_path = os.path.join(FEEDBACK_DIR, f"low_conf_{timestamp}.jpg")
                            cv2.imwrite(file_path, frame)
                            print(f"✅ Feedback Saved: Confidence={min_conf:.2f} to {file_path}")
                except Exception as e:
                    print(f"⚠️ Feedback logic error: {e}")

            ret, buffer = cv2.imencode(".jpg", annotated_frame)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    except GeneratorExit:
        pass
    except Exception as e:
        print(f"🔴 ERROR: stream loop crashed: {e}")
        print(traceback.format_exc())
    finally:
        try:
            cap.release()
        except Exception:
            pass

@app.get("/video")
def video_feed():
    return StreamingResponse(get_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/health")
def health():
    try:
        exists = os.path.exists(MODEL_PATH)
        return JSONResponse(status_code=200, content={"status": "ok", "model_path": MODEL_PATH, "model_exists": exists})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

