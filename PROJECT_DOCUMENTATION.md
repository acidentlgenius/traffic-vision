# Project Breakdown: Traffic Vision (PoC)

**Note**: This isn't just a corporate design document. It's a breakdown of the specific engineering challenges I faced and how I solved them.

---

## 1. Executive Summary
**Traffic Vision** is a production-grade MLOps platform designed for autonomous vehicle perception.
The core goal: Build a system that doesn't just fail when the environment changes (e.g., day to night), but **adapts**.

It demonstrates a complete end-to-end MLOps lifecycle, moving beyond simple model training to address critical production challenges:
1.  **Drift Detection**: Catching failures before they happen.
2.  **Automated Recovery**: Retraining without human intervention.
3.  **Zero-Downtime Deployment**: Updating the model while the car is driving.

---

## 2. System Architecture

The architecture decouples the "Fast Path" (Inference) from the "Slow Path" (Training).

### High-Level Components
1.  **DataOps & Active Learning Simulation**:
    -   **Dataset**: Uses BDD100K (split into Day/Night).
    -   **Drift Loop**: Identifies "low confidence" or drifted data.
    -   **Simulation**: Instead of waiting for human labelers, I simulate a "Labeling Service" that fetches ground truth for drifted samples.
2.  **Orchestration (Airflow)**:
    -   Manages the complex graph: *Ingest -> Purity Check -> Train -> Validate -> Deploy*.
3.  **Serving (FastAPI + ONNX)**:
    -   A specialized inference server designed for hot-swapping.

---

## 3. MLOps Technology Stack

I chose this stack to balance "Industry Standard" with "Prototype Speed".

| Component | Tool | Why I Chose It |
| :--- | :--- | :--- |
| **Data Versioning** | **DVC** | Git chokes on large binaries. DVC lets me `checkout` the exact 10GB dataset used for Model V1. |
| **Orchestration** | **Apache Airflow** | Scripts are brittle. Airflow handles retries and dependencies (e.g., ensuring the DB is up before training starts). |
| **Serving** | **FastAPI + ONNX** | **FastAPI** gives me async performance. **ONNX** makes the model portable across hardware. |
| **Monitoring** | **Statistical Checks** | I built a custom drift monitor using **KS-Tests** (Scipy) because it's lighter than deploying a full monitoring server for a PoC. |
| **Containerization** | **Docker** | Guarantees that "It works on my machine" means "It works on yours too." |

---

## 4. Key Implementation Details

### 🔄 The Self-Correcting Loop
The defining feature of Traffic Vision is its ability to adapt.
1.  **Baseline**: A YOLOv8 Nano model is trained on "Day" data.
2.  **Drift Injection**: "Night" data is fed to the inference service.
3.  **Detection**: The statistical monitor detects the distribution shift.
4.  **Trigger**: An Airflow DAG is triggered. It augments the training set with the "Night" data.
5.  **Hot-Swap**: The pipeline triggers the `/reload` endpoint on the API.
6.  **Result**: A V2 model is deployed that handles night driving, **without the container ever restarting**.

### 🚀 CI/CD & Automation
I separated the concerns:
-   **GitHub Actions**: Checks **Code Quality** (Linting, Unit Tests).
-   **Airflow**: Checks **Data Quality** (Drift, Schema validation).

---

## 5. Alignment with MLOps Competencies

This project was built to demonstrate specific core competencies:

| MLOps Competency | Project Implementation |
| :--- | :--- |
| **Pipelines** | Architected a modular system using **Docker Compose** to orchestrate Airflow and API services. |
| **Monitoring** | Implemented real-time statistical drift monitoring that closes the feedback loop. |
| **Reproducibility** | Integrated **DVC** and **MLflow** so that every model artifact can be traced back to the exact code and data commit. |
| **Reliability** | Built the **Hot-Swap** mechanism to ensure zero downtime during updates—critical for AV safety. |

---

## 6. What would I do differently in Production?
This is a robust prototype, but here is what I would change for enterprise scale:
-   **Kubernetes**: Move from Docker Compose to K8s for true auto-scaling.
-   **Real Active Learning**: Swap the "Simulated Labeling" for a tool like **Label Studio** or **Scale AI**.
-   **Shadow Mode**: Run the V2 model in "Shadow Mode" (process requests but don't return answers) for 1 hour before full promotion.
