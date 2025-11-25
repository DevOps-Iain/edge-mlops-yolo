# src/app.py
from ultralytics import YOLO
import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from datetime import datetime
from functools import lru_cache
import os
import time

app = FastAPI()

# 以 app.py 所在位置為基準，推算專案根目錄
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# 預設模型與回饋資料夾路徑：
# - 在本機執行：會是「專案根目錄 / best.onnx」與「專案根目錄 / feedback_data」
# - 在 Docker 容器：會被環境變數 MODEL_PATH、FEEDBACK_DIR 覆蓋成 /app/...
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(PROJECT_ROOT, "best.onnx"))
FEEDBACK_DIR = os.getenv("FEEDBACK_DIR", os.path.join(PROJECT_ROOT, "feedback_data"))

os.makedirs(FEEDBACK_DIR, exist_ok=True)
print(f"Feedback directory initialized: {FEEDBACK_DIR}")
print(f"Model path: {MODEL_PATH}")

from functools import lru_cache

@lru_cache(maxsize=1)
def get_model():
    return YOLO(MODEL_PATH, task="detect")

@lru_cache(maxsize=1)
def get_model():
    # Lazy load 模型
    return YOLO(MODEL_PATH, task="detect")

def get_frames():
    cap = cv2.VideoCapture(0)

    # 設定解析度（之後 Pi 上用得到）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("🔴 ERROR: Camera not accessible. Check /dev/video0 permission.")
        time.sleep(5)
        return

    model = get_model()

    while True:
        success, frame = cap.read()
        if not success:
            print("🔴 ERROR: Failed to read frame.")
            break

        results = model(frame, stream=True, verbose=False)

        for result in results:
            annotated_frame = result.plot()

            # --- MLOps 數據回饋邏輯 ---
            try:
                if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
                    min_conf = float(result.boxes.conf.min().item())
                    if 0.25 < min_conf < 0.5:
                        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                        file_path = os.path.join(FEEDBACK_DIR, f"low_conf_{timestamp}.jpg")
                        cv2.imwrite(file_path, frame)
                        print(f"✅ Feedback Saved: Confidence={min_conf:.2f} to {file_path}")
            except Exception as e:
                print(f"⚠️ Feedback logic error: {e}")

            # 轉碼成 JPEG 並串流回傳
            ret, buffer = cv2.imencode(".jpg", annotated_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

@app.get("/video")
def video_feed():
    return StreamingResponse(
        get_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/health")
def health():
    """
    健康檢查，用於 Docker healthcheck 或外部監控。
    """
    try:
        exists = os.path.exists(MODEL_PATH)
        return JSONResponse(
            status_code=200,
            content={"status": "ok", "model_path": MODEL_PATH, "model_exists": exists}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

