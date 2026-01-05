from ultralytics import YOLO
import shutil
import os
import requests
from pathlib import Path

# Paths
RAW_DIR = Path("data/raw")
# TRAIN_DIR not used directly as global anymore, specific functions manage subdirs
DRIFT_DIR = RAW_DIR / "images/drift"
DRIFT_DIR.mkdir(parents=True, exist_ok=True)

def download_coco128():
    """Downloads COCO128 and moves car/bus/truck images to training set."""
    print("Downloading COCO128...")
    # This downloads to current dir/datasets/coco128 usually
    # We can just load the model and it will trigger download if needed, or use util
    from ultralytics.utils.downloads import download
    
    url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip'
    download(url, dir='.')
    
    # Move images & labels
    src_img_dir = Path('coco128/images/train2017')
    src_lbl_dir = Path('coco128/labels/train2017')
    
    # Destination structure for YOLO
    # data/raw/images/train
    # data/raw/labels/train
    
    dest_img_dir = Path("data/raw/images/train")
    dest_lbl_dir = Path("data/raw/labels/train")
    
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)

    if src_img_dir.exists():
        for img in src_img_dir.glob("*.jpg"):
            shutil.copy(img, dest_img_dir / img.name)
            # Copy corresponding label
            label_name = img.stem + ".txt"
            label_path = src_lbl_dir / label_name
            if label_path.exists():
                shutil.copy(label_path, dest_lbl_dir / label_name)
        
    # Clean up
    shutil.rmtree('coco128', ignore_errors=True)
    print(f"Moved images/labels to {dest_img_dir} and {dest_lbl_dir}")

def download_night_samples():
    """Downloads a few night time street scenes to simulate drift."""
    print("Downloading night samples...")
    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/b/bb/Nachtverkehr_auf_der_Autobahn_A_5_bei_Frankfurt_am_Main_img_010.jpg", 
        "https://upload.wikimedia.org/wikipedia/commons/e/e5/Night_traffic_in_Bangkok.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/8/82/Tokyo_Night_Street.jpg"
    ]
    
    for i, url in enumerate(urls):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                with open(DRIFT_DIR / f"night_{i}.jpg", "wb") as f:
                    f.write(resp.content)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            
    print(f"Downloaded {len(list(DRIFT_DIR.glob('*')))} night images.")

if __name__ == "__main__":
    download_coco128()
    download_night_samples()
