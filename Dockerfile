FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements-docker.txt \
    && python -m pip install \
       https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

COPY docker/patch_deepwalk.py /tmp/patch_deepwalk.py
RUN python /tmp/patch_deepwalk.py && rm /tmp/patch_deepwalk.py

WORKDIR /workspace/HeterSGT
COPY . .

CMD ["python", "main.py", "--help"]
