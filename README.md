# AI Project Template

This repository contains the core, hardware-independent code for a config-driven AI pipeline project. It includes the source files, configurations, data, and tests.

**For environment setup and infrastructure instructions (including Docker, RunPod, etc.), please see the dedicated infrastructure repository.**

## Quickstart

Once your development environment is configured (e.g., inside a Docker container from the infrastructure repository), you can run the full pipeline.

The main training pipeline can be executed with a single script, which takes a configuration file as an argument:

```bash
# Run the full pipeline using the baseline experiment configuration
bash scripts/run_train.sh configs/exp_baseline.yaml
```

This script will handle:
1.  Preprocessing the data
2.  Generating features
3.  Training the model
4.  Evaluating the results

## Environment Setup

- Script: `AI_Template/scripts/bootstrap_env.sh`
- Purpose: Installs Python (optional), sets up a virtualenv, installs project dependencies, and optionally installs Quarto and TeX for PDF/Quarto rendering.

Examples (run from repo root):

```bash
# Standard setup: venv + deps + Quarto + BasicTeX (smaller)
bash AI_Template/scripts/bootstrap_env.sh

# Use a specific Python version and full TeX (large download)
bash AI_Template/scripts/bootstrap_env.sh --python-version 3.11 --tex full

# Skip system package installs (brew/apt) if you manage them yourself
bash AI_Template/scripts/bootstrap_env.sh --no-system
```

Notes:
- macOS: Uses Homebrew to install Python, Quarto, and either BasicTeX (default) or full MacTeX.
- Linux (Debian/Ubuntu): Uses apt for Python and TeX; downloads Quarto `.deb` if selected.
- After completion, activate the environment with `source .venv/bin/activate` (or your chosen `--venv`).


## Project Structure

- `src/yourproj`: Main source code for the project.
- `configs`: YAML configuration files for different experiments.
- `data`: Raw and processed data.
- `notebooks`: Jupyter notebooks for exploration and analysis.
- `scripts`: Helper scripts for running parts of the pipeline.
- `tests`: Unit and integration tests.
- `report`: LaTeX files for generating reports.
