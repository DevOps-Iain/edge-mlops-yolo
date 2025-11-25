# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安裝必要的系統底層依賴 (解決 OpenCV 在 Linux 上的缺套件問題)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*


# 安裝 Python 套件
RUN pip install --no-cache-dir ultralytics fastapi "uvicorn[standard]" numpy opencv-python

# 複製程式碼與模型
COPY src/app.py /app/
COPY best.onnx /app/
RUN mkdir -p /app/feedback_data

# 環境變數（可被 docker-compose.yml 覆蓋）
ENV MODEL_PATH=/app/best.onnx
ENV FEEDBACK_DIR=/app/feedback_data

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


