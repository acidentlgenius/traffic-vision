FROM python:3.12-slim

WORKDIR /app

# Install essentials only if needed (usually none for pure python wheels)
# We switched to opencv-headless so no libgl1 needed.
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to reduce image size and build time
# (Ultralytics would otherwise pull the huge CUDA version)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# Increase timeout for slow networks, use cache
RUN pip install --no-cache-dir -r requirements.txt

# Ensure we use headless opencv (uninstall the one pulled by ultralytics if any)
RUN pip uninstall -y opencv-python || true
RUN pip install --force-reinstall opencv-python-headless



COPY src/app.py src/
# Copy the model we verified exists locally
COPY models/best.onnx models/best.onnx

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
