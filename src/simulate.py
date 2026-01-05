import requests
import time
import os
from pathlib import Path

API_URL = "http://127.0.0.1:8000/predict"
# Use the images we downloaded earlier
DAY_IMAGES = list(Path("data/raw/images/train").glob("*.jpg"))[:5]
# Night images
NIGHT_IMAGES = list(Path("data/raw/images/drift").glob("*.jpg"))

def send_request(img_path):
    try:
        with open(img_path, "rb") as f:
            files = {"file": f}
            resp = requests.post(API_URL, files=files)
            print(f"Sent {img_path.name}: {resp.status_code}")
            if resp.status_code == 200:
                print(resp.json())
    except Exception as e:
        print(f"Error sending {img_path.name}: {e}")

def simulate():
    print("Simulating DAY traffic (Baseline)...")
    for img in DAY_IMAGES:
        send_request(img)
        time.sleep(0.5)
        
    print("\nSimulating NIGHT traffic (Drift)...")
    for img in NIGHT_IMAGES:
        send_request(img)
        time.sleep(0.5)

if __name__ == "__main__":
    if not (DAY_IMAGES or NIGHT_IMAGES):
        print("No images found to simulate.")
        exit(1)
    simulate()
