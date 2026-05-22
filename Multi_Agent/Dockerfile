# CUDA + cuDNN + PyTorch preinstalled. Pin to a known-good combo.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Python deps first so they cache independently of source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source + datasets.
COPY . .

# Default: full experiment. Override in docker-compose or `docker run ... <args>`.
ENTRYPOINT ["python", "main.py"]
