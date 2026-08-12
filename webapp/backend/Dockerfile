FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first so the image does not pull CUDA wheels.
# The CPU index has its own available versions, so I do not pin these here.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Then install the remaining deps. torch/torchvision lines in requirements.txt
# are already satisfied, so pip will skip them.
COPY webapp/backend/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# The ML code is also mounted by docker-compose during local development.
# We also copy as a fallback so `docker run` without mounts still works.
COPY src /app/src
COPY webapp/backend/app /app/app

ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
