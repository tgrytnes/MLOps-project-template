# AI Project Template — Local + Cloud Ready

**Source files-first** , **config-driven pipelines**, stable **artifacts** for LaTeX, tests, pre-commit, and CI.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
pre-commit install
make smoke
jupyter lab
make pipeline CONFIG=configs/exp_baseline.yaml
make report   # builds report/out/main.pdf
```

## Installation Profiles

- CPU (lightweight):
  ```bash
  pip install -U pip && pip install .
  ```

- GPU (heavy, PyTorch):
  - For CUDA wheels (recommended on GPU hosts):
    ```bash
    pip install -U pip
    pip install '.[gpu]' --index-url https://download.pytorch.org/whl/cu121
    ```
  - For CPU-only wheels (works everywhere but no GPU acceleration):
    ```bash
    pip install -U pip && pip install '.[gpu]'
    ```

- TensorFlow (CPU):
  ```bash
  pip install -U pip && pip install '.[tf]'
  ```

- TensorFlow (GPU, Linux x86_64):
  ```bash
  pip install -U pip && pip install '.[tf-gpu]'
  # Requires NVIDIA Container Toolkit and compatible host drivers
  ```

## Cloud
```bash
make docker-build
# on a GPU host with nvidia-container-toolkit:
docker run --gpus all -v $(pwd):/work -w /work yourproj:latest   bash -lc "python -m yourproj.train --config configs/exp_baseline.yaml"
```

### Docker images

- CPU (base):
  ```bash
  make docker-build
  docker run -it --rm -v $(pwd):/work -w /work yourproj:latest bash
  ```

- PyTorch GPU:
  ```bash
  make docker-build-gpu
  docker run --gpus all -it --rm -v $(pwd):/work -w /work yourproj:gpu bash
  ```

- TensorFlow CPU:
  ```bash
  make docker-build-tf
  docker run -it --rm -v $(pwd):/work -w /work yourproj:tf bash
  ```

- TensorFlow GPU:
  ```bash
  make docker-build-tf-gpu
  docker run --gpus all -it --rm -v $(pwd):/work -w /work yourproj:tf-gpu bash
  ```

Notes:
- TensorFlow GPU via pip (`tensorflow[and-cuda]`) is currently supported on Linux x86_64. The container still requires NVIDIA drivers on the host.
- Raspberry Pi: TensorFlow wheels may be limited; PyTorch CPU or pure-NumPy/scikit-learn workloads are more feasible on Pi.

### docker compose profiles

Interactive shells with the compose profiles defined in `compose.yaml`:

```bash
# CPU
docker compose --profile cpu run --rm cpu bash

# PyTorch GPU (requires NVIDIA Container Toolkit)
docker compose --profile gpu run --rm --gpus all gpu bash

# TensorFlow CPU
docker compose --profile tf run --rm tf bash

# TensorFlow GPU (requires NVIDIA Container Toolkit)
docker compose --profile tf-gpu run --rm --gpus all tf-gpu bash
```

JupyterLab services under each profile:

```bash
# CPU JupyterLab on http://localhost:8888
make compose-cpu-lab

# PyTorch GPU JupyterLab on http://localhost:8889 (requires NVIDIA runtime)
make compose-gpu-lab

# TensorFlow CPU JupyterLab on http://localhost:8890
make compose-tf-lab

# TensorFlow GPU JupyterLab on http://localhost:8891 (requires NVIDIA runtime)
make compose-tf-gpu-lab
```

All JupyterLab instances start without token/password inside the container for local development convenience. Consider adding auth or `-d` (detached) if exposing ports more broadly.
