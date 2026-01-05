# Traffic Vision - Self-Correcting Object Detection Pipeline

This project implements a production-grade MLOps pipeline for autonomous vehicle object detection. 
It features a "monitoring-first" architecture that detects environmental drift (e.g., day vs. night) and triggers simulated active learning workflows.

## Tech Stack
*   **Data:** BDD100K
*   **Version Control:** DVC + Git
*   **Model:** YOLOv8 (ONNX optimized)
*   **Orchestration:** Airflow
*   **Monitoring:** Evidently AI
