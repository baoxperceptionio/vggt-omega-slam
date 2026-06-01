# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE_IMAGE}

ARG VGGT_OMEGA_CHECKPOINT_DIR=/app/checkpoints/VGGT-Omega-1B-512
ARG VGGT_OMEGA_TEXT_CHECKPOINT_DIR=/app/checkpoints/VGGT-Omega-1B-256-Text-Alignment

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VGGT_OMEGA_CHECKPOINT=${VGGT_OMEGA_CHECKPOINT_DIR}/model.pt \
    VGGT_OMEGA_TEXT_CHECKPOINT=${VGGT_OMEGA_TEXT_CHECKPOINT_DIR}/model.pt \
    VGGT_OMEGA_IMAGE_RESOLUTION=512

WORKDIR /app

COPY requirements.txt requirements_demo.txt pyproject.toml README.md LICENSE ./
COPY vggt_omega ./vggt_omega
COPY demo_gradio.py visual_util.py ./
COPY scripts ./scripts
COPY examples ./examples

RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt -r requirements_demo.txt && \
    python -m pip install -e .

RUN --mount=type=secret,id=hf_token,required=true <<"EOF"
set -eu
python - <<"PY"
from pathlib import Path
import os
import shutil
import urllib.request

token = Path("/run/secrets/hf_token").read_text(encoding="utf-8").strip()
downloads = {
    "vggt_omega_1b_512.pt": Path(os.environ["VGGT_OMEGA_CHECKPOINT"]),
    "vggt_omega_1b_256_text.pt": Path(os.environ["VGGT_OMEGA_TEXT_CHECKPOINT"]),
}

for checkpoint_file, checkpoint_path in downloads.items():
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/facebook/VGGT-Omega/resolve/main/{checkpoint_file}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request) as response, checkpoint_path.open("wb") as output:
        shutil.copyfileobj(response, output)
PY
EOF

EXPOSE 7860

CMD ["python", "demo_gradio.py", "--checkpoint", "/app/checkpoints/VGGT-Omega-1B-512/model.pt", "--image-resolution", "512", "--server-name", "0.0.0.0", "--server-port", "7860"]
