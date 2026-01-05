from ultralytics import YOLO
import mlflow
import os
from pathlib import Path
from ultralytics import settings

# Disable Ultralytics auto-logging to MLflow to avoid conflicts with our explicit control
settings.update({"mlflow": False, "wandb": False})

# Paths
DATA_DIR = Path("data/raw")
TRAIN_DIR = DATA_DIR / "day"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def train_model():
    """Trains a YOLOv8 Nano model on the dataset and logs to MLflow."""
    
    # Initialize MLflow
    mlflow.set_experiment("traffic-vision-yolo")
    
    with mlflow.start_run():
        # Log parameters
        params = {
            "model": "yolov8n.pt",
            "epochs": 5,  # Short epochs for MVP
            "imgsz": 640,
            "batch": 8
        }
        mlflow.log_params(params)
        
        # Initialize model
        model = YOLO(params["model"])
        
        # Train
        # Note: YOLOv8 usually expects a data.yaml file. We need to create one dynamically or use the folder structure if supported.
        # Update data.yaml to point to new structure
        # YOLO expects 'path' to be root, and train/val relative to it OR 'train' as absolute path to images
        # Simplest is:
        # train: /abs/path/to/data/raw/images/train
        # val: /abs/path/to/data/raw/images/train (same for mvp)
        
        base_path = os.path.abspath("data/raw")
        
        yaml_content = f"""
        path: {base_path}
        train: images/train
        val: images/train
        names:
          0: person
          1: bicycle
          2: car
          3: motorcycle
          4: airplane
          5: bus
          6: train
          7: truck
          8: boat
          9: traffic light
        """
        
        with open("data.yaml", "w") as f:
            f.write(yaml_content)
            
        print("Starting training...")
        results = model.train(
            data="data.yaml",
            epochs=params["epochs"],
            imgsz=params["imgsz"],
            batch=params["batch"],
            project="mlruns",
            name="yolov8n_experiment"
        )
        
        # Log metrics
        # Results object has metrics, but MLflow autologging with Ultralytics is tricky sometimes.
        # We'll log the final mAP manually if accessible, or just rely on artifacts.
        mlflow.log_metric("map50", results.box.map50)
        
        # Export to ONNX
        success = model.export(format="onnx")
        print(f"Export success: {success}")
        
        # Log the ONNX model artifact
        onnx_path = list(Path(results.save_dir).glob("*.onnx"))
        if onnx_path:
            mlflow.log_artifact(str(onnx_path[0]), artifact_path="model")
            # Also copy to our models dir for easy access by the serving app
            import shutil
            shutil.copy(onnx_path[0], MODELS_DIR / "best.onnx")

if __name__ == "__main__":
    train_model()
