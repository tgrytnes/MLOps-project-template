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

## Project Structure

- `src/yourproj`: Main source code for the project.
- `configs`: YAML configuration files for different experiments.
- `data`: Raw and processed data.
- `notebooks`: Jupyter notebooks for exploration and analysis.
- `scripts`: Helper scripts for running parts of the pipeline.
- `tests`: Unit and integration tests.
- `report`: LaTeX files for generating reports.

