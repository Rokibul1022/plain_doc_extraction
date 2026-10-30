# syntax=docker/dockerfile:1
# Whole web app (plate UI with Image / Video / OpenCV tabs) in one image. Prefer the two focused images for production:
#   Dockerfile.opencv  - OpenCV-only scan service        Dockerfile.ocr  - YOLO + OCR service
#
#   docker build -t plate-app .
#   docker run --rm -p 8000:8000 plate-app
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    OPENCV_THREADS=1 ORT_INTRA_THREADS=2 ORT_INTER_THREADS=1 \
    HOME=/home/app

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

WORKDIR /app
COPY requirements.txt /tmp/requirements-all.txt
COPY opencv/requirements.txt /tmp/opencv/requirements.txt
COPY ocr/requirements.txt /tmp/ocr/requirements.txt
RUN grep -v -E "^(pytest|httpx|-r )" /tmp/requirements-all.txt > /tmp/app-requirements.txt \
    && sed -i 's#^-r ../opencv/requirements.txt#-r /tmp/opencv/requirements.txt#' /tmp/ocr/requirements.txt \
    && pip install -r /tmp/opencv/requirements.txt -r /tmp/ocr/requirements.txt -r /tmp/app-requirements.txt

COPY --chown=app:app app ./app
COPY --chown=app:app opencv ./opencv
COPY --chown=app:app ocr ./ocr
COPY --chown=app:app ocr_merge ./ocr_merge
COPY --chown=app:app run.py ./run.py
USER app

# bake the YOLO plate detector + plate OCR model into the image
RUN python -c "\
from open_image_models import create_detector; create_detector('yolo-v9-t-640-license-plate-end2end', providers=['CPUExecutionProvider']); \
from fast_plate_ocr import LicensePlateRecognizer; LicensePlateRecognizer('cct-xs-v1-global-model', device='cpu')"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/health', timeout=8)"

CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8000", "--no-browser"]
