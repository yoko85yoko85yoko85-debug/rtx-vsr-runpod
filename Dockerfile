FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg python3-venv libgomp1 && rm -rf /var/lib/apt/lists/*
RUN python -m venv --system-site-packages /opt/vsr-venv
ENV PATH="/opt/vsr-venv/bin:${PATH}"
COPY requirements.txt /tmp/vsr-requirements.txt
RUN python -c "import torch; print('torch=='+torch.__version__.split('+')[0])" > /tmp/torch-constraint.txt && python -m pip install -c /tmp/torch-constraint.txt --extra-index-url https://pypi.nvidia.com -r /tmp/vsr-requirements.txt
WORKDIR /app
COPY handler.py runtime.py media.py vsr_engine.py ./
RUN python -m py_compile handler.py runtime.py media.py vsr_engine.py && python -c "import sys, torch, importlib.util, importlib.metadata; assert sys.prefix != sys.base_prefix; assert int(torch.version.cuda.split('.')[0]) >= 13; assert importlib.util.find_spec('nvvfx') is not None; assert importlib.metadata.version('nvidia-vfx') == '0.2.0.0'; print('BUILD_PACKAGE_OK: CUDA13 and SDK installed; driver/SDK import and inference checked at runtime')"
CMD ["python", "-u", "handler.py"]
