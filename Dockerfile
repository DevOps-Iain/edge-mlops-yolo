FROM python:3.9-slim

WORKDIR /app

# 建議用 headless 版 OpenCV，容器更穩
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" numpy opencv-python-headless \
    ultralytics \
    onnx==1.16.2 onnxruntime==1.19.2

ENV YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PYTHONUNBUFFERED=1

COPY src/app.py /app/app.py
COPY best.onnx /app/best.onnx
RUN mkdir -p /app/feedback_data /tmp/Ultralytics

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
