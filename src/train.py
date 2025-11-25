# src/train.py
from ultralytics import YOLO
import mlflow
import os
import shutil
import yaml
from datetime import datetime

# 取得 config 檔路徑
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "train.yaml")

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_git_commit():
    try:
        return os.popen("git rev-parse HEAD").read().strip()
    except Exception:
        return "unknown"

def train_and_export():
    cfg = load_config(CONFIG_PATH)

    model_name = cfg["model_name"]
    data_path = cfg["data_path"]
    epochs = cfg["epochs"]
    img_size = cfg["img_size"]
    device = cfg["device"]
    experiment_name = cfg["experiment_name"]

    # --- MLflow 設定 ---
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)

    run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        # 1. 記錄參數與環境資訊
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("img_size", img_size)
        mlflow.log_param("device", device)

        git_commit = get_git_commit()
        mlflow.set_tag("git_commit", git_commit)

        # 2. 載入並訓練模型
        model = YOLO(model_name)
        results = model.train(
            data=data_path,
            epochs=epochs,
            imgsz=img_size,
            device=device
        )

        # 3. 記錄關鍵指標
        mAP50 = 0.0
        try:
            mAP50 = results.metrics.get("metrics/mAP50(B)", 0.0)
        except Exception:
            pass

        mlflow.log_metric("mAP50", float(mAP50))

        # 4. 匯出 ONNX（給 Pi 使用）
        export_path = model.export(format="onnx", dynamic=False)
        onnx_target = "./best.onnx"
        shutil.copy(export_path, onnx_target)

        # 5. 記錄模型 & 設定檔為 artifacts
        mlflow.log_artifact(onnx_target, artifact_path="model")
        mlflow.log_artifact(CONFIG_PATH, artifact_path="config")

        print(f"✅ Training completed. ONNX saved to {onnx_target}")

if __name__ == "__main__":
    train_and_export()

