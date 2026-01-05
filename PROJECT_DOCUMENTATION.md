# Traffic Vision: Self-Correcting Object Detection Pipeline for Autonomous Vehicles

## 1. Executive Summary
**Traffic Vision** is a production-grade MLOps platform designed for autonomous vehicle perception. The core objective is to build a robust object detection system that not only performs real-time inference but also includes a "self-correcting" mechanism. The system continuously monitors for environmental data drift (specifically changes in lighting conditions like day vs. night) and automatically triggers simulated active learning workflows to retrain and update the model, ensuring consistent performance in dynamic real-world scenarios.

This project demonstrates a complete end-to-end MLOps lifecycle, moving beyond simple model training to address critical production challenges: reproducibility, automation, monitoring, and continuous improvement.

## 2. System Architecture

The architecture is designed to decouple concerns between data operations, model training, and application serving, bonded together by robust orchestration and CI/CD pipelines.

### High-Level Components
1.  **DataOps & Active Learning Simulation**:
    -   **Dataset**: Uses BDD100K (split into Day/Night) to simulate domain shift.
    -   **Drift Loop**: The system identifies "low confidence" or drifted data (e.g., night scenes) during inference.
    -   **Simulation**: Instead of manual labeling, the system fetches ground truth for these drifted samples from a held-out dataset, simulating a human-in-the-loop active learning process.
2.  **Orchestration (Airflow)**:
    -   Manages long-running data and model pipelines. Dags are triggered by drift alarms to ingest new data, retrain the model, and register the new version.
3.  **CI/CD (GitHub Actions)**:
    -   Ensures software quality with every commit. Pipelines run linting, unit tests, and build Docker images for the inference service.
4.  **Serving (FastAPI + ONNX)**:
    -   Provides a lightweight, low-latency REST API for model inference, optimized for edge-like deployment scenarios using ONNX Runtime.

## 3. MLOps Technology Stack

This project utilizes a "All-Free" open-source stack chosen for industry standard relevance and efficiency:

| Component | Tool | Justification |
| :--- | :--- | :--- |
| **Data Versioning** | **DVC (Data Version Control)** | Manages large datasets (BDD100K) and ensures data reproducibility alongside code in Git. |
| **Experiment Tracking** | **MLflow** | Tracks hyperparameters, metrics, and manages model artifact versioning (Model Registry). |
| **Orchestration** | **Apache Airflow** | Handles complex dependencies for retraining workflows (Data Ingest -> Train -> Validate -> Deploy). |
| **CI/CD** | **GitHub Actions** | Automates the software lifecycle: linting (`flake8`), testing (`pytest`), and container delivery. |
| **Serving** | **FastAPI + ONNX** | **FastAPI** provides modern, async high-performance API capabilities. **ONNX** ensures framework-agnostic, optimized inference. |
| **Monitoring** | **Evidently AI** | specialized tool for detecting data drift (comparing training vs. production data distributions) to trigger retraining. |
| **Containerization** | **Docker & Docker Compose** | Ensures environment consistency across development, testing, and production (simulated) environments. |

## 4. Key Implementation Details

### 🔄 The Self-Correcting Loop
The defining feature of Traffic Vision is its ability to adapt.
1.  **Baseline**: A YOLOv8 Nano model is trained on "Day" data.
2.  **Drift Injection**: "Night" data is fed to the inference service to simulate environmental change.
3.  **Detection**: **Evidently AI** detects the distribution shift in image characteristics and model confidence.
4.  **Trigger**: An Airflow DAG is triggered automatically. It augments the training set with the "Night" data (simulated labeling) and retrains the model.
5.  **Hot-Swap**: The pipeline triggers a zero-downtime reload of the inference service via a specialized API endpoint.
6.  **Result**: A V2 model is deployed that creates accurate detections in both day and night conditions, without restarting containers.

### 🚀 CI/CD Implementation
We strictly separate "Code Validation" from "Model Training":
-   **GitHub Actions**: Runs on git push. Focuses on code correctness—linting, type checking, and unit tests for the API code. Builds the serving container.
-   **Airflow**: Runs on schedule or trigger. Focuses on data/model correctness—fetching data, heavy training jobs, and model evaluation.

## 5. Alignment with MLOps Competencies

This project serves as a practical demonstration of key MLOps skills:

| MLOps Competency | Project Implementation |
| :--- | :--- |
| **Design & maintain ML infrastructure and pipelines** | Architected a modular system using **Docker Compose** to orchestrate Airflow, MLflow, and API services locally, simulating a distributed production environment. |
| **Automate deployment & monitoring** | Implemented **GitHub Actions** for CI/CD of the serving app and **Evidently AI** for real-time performance and drift monitoring. |
| **Model versioning & reproducibility** | integrated **DVC** for dataset versioning and **MLflow** for tracking experiments and registering model versions, ensuring every result is reproducible. |
| **Collaborate to transition prototypes to production** | Converted experimental YOLOv8 PyTorch models into optimized **ONNX** format wrapped in a **FastAPI** service with **hot-swap capabilities** for zero-downtime updates. |
| **Troubleshoot & improve reliability** | Designed the system to be resilient to data drift (a common production failure mode) by building the automated retraining feedback loop, capable of **self-healing** in production. |
| **Ensure security & scalability** | Used containerization (**Docker**) to isolate services and standard API interfaces, laying the groundwork for Kubernetes-based scaling. |

## 6. Future Roadmap
-   **Kubernetes Deployment**: Move from Docker Compose to a K8s cluster for true auto-scaling.
-   **Real Active Learning**: Replace the simulation with a Label Studio integration for actual human-in-the-loop labeling.
-   **Model A/B Testing**: Implement distinct endpoints to route traffic between Model V1 and V2 to measure real-world performance differences.
