export PYTHONPATH := src
SHELL := /bin/bash
export PYTHONPATH := src
PY := python
CONFIG ?= configs/exp_baseline.yaml

.PHONY: env install install-cpu install-gpu install-tf install-tf-gpu lint test smoke eda pipeline train eval report docker-build docker-build-gpu docker-build-tf docker-build-tf-gpu compose-cpu compose-gpu compose-tf compose-tf-gpu compose-cpu-lab compose-gpu-lab compose-tf-lab compose-tf-gpu-lab clean sync_snippets

env:
	python -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -r requirements.txt && pre-commit install

install:
	pip install -r requirements.txt

# Optional installs via pyproject extras (Option A)
install-cpu:
	pip install -U pip && pip install .

install-gpu:
	@echo "Installing GPU extras (PyTorch). If you want CUDA wheels, add: --index-url https://download.pytorch.org/whl/cu121"
	pip install -U pip && pip install '.[gpu]'

install-tf:
	pip install -U pip && pip install '.[tf]'

install-tf-gpu:
	@echo "Installing TensorFlow with CUDA libs via pip (Linux x86_64). Requires NVIDIA Container Toolkit on host."
	pip install -U pip && pip install '.[tf-gpu]'

lint:
	ruff check --fix .
	black .

test:
	pytest -q

smoke:
	$(PY) -m yourproj.smoke $(CONFIG)

eda:
	@echo "Open notebooks/main.ipynb in Jupyter to explore."

sync_snippets:
	python3 scripts/refresh_snippets.py
	@echo "Embedded src snippets refreshed in-place (no jupytext needed)."

pipeline:
	$(PY) -m yourproj.preprocess $(CONFIG)
	$(PY) -m yourproj.features $(CONFIG)
	$(PY) -m yourproj.train $(CONFIG)
	$(PY) -m yourproj.eval $(CONFIG)

train:
	$(PY) -m yourproj.train $(CONFIG)

eval:
	$(PY) -m yourproj.eval $(CONFIG)

report:
	@echo "Building LaTeX report..."
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=out main.tex
	@echo "Report built at report/out/main.pdf"

docker-build:
	docker build -t yourproj:latest -f Dockerfile .

docker-build-gpu:
	docker build -t yourproj:gpu -f Dockerfile.gpu .

docker-build-tf:
	docker build -t yourproj:tf -f Dockerfile.tf .

docker-build-tf-gpu:
	docker build -t yourproj:tf-gpu -f Dockerfile.tf-gpu .

clean:
	rm -rf report/out

# Convenience compose runners (interactive shells)
compose-cpu:
	docker compose --profile cpu run --rm cpu bash

compose-gpu:
	docker compose --profile gpu run --rm --gpus all gpu bash

compose-tf:
	docker compose --profile tf run --rm tf bash

compose-tf-gpu:
	docker compose --profile tf-gpu run --rm --gpus all tf-gpu bash

compose-cpu-lab:
	docker compose --profile cpu up --build cpu-lab

compose-gpu-lab:
	docker compose --profile gpu up --build --quiet-pull gpu-lab

compose-tf-lab:
	docker compose --profile tf up --build tf-lab

compose-tf-gpu-lab:
	docker compose --profile tf-gpu up --build --quiet-pull tf-gpu-lab
