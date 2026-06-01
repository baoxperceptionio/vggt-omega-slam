# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE_IMAGE}

ARG VGGT_OMEGA_CHECKPOINT_DIR=/opt/vggt-omega/checkpoints/VGGT-Omega-1B-512
ARG VGGT_OMEGA_TEXT_CHECKPOINT_DIR=/opt/vggt-omega/checkpoints/VGGT-Omega-1B-256-Text-Alignment

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VGGT_OMEGA_CHECKPOINT=${VGGT_OMEGA_CHECKPOINT_DIR}/model.pt \
    VGGT_OMEGA_TEXT_CHECKPOINT=${VGGT_OMEGA_TEXT_CHECKPOINT_DIR}/model.pt \
    VGGT_OMEGA_IMAGE_RESOLUTION=512

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements.txt requirements_demo.txt /tmp/vggt-omega-build/

RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/vggt-omega-build/requirements.txt -r /tmp/vggt-omega-build/requirements_demo.txt

RUN --mount=type=secret,id=hf_token,required=false <<"EOF"
set -eu
python - <<"PY"
from pathlib import Path
import os
import shutil
import urllib.request

token_path = Path("/run/secrets/hf_token")
if not token_path.exists() or not token_path.read_text(encoding="utf-8").strip():
    print("No hf_token build secret provided; skipping checkpoint download.")
    raise SystemExit(0)

token = token_path.read_text(encoding="utf-8").strip()
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
