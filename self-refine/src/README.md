# Self-Refine Experiments

This repository contains the implementation and results for the **Self-Refine** experiments, including:
- the main pipeline (`self_refine.py`) and (`dataset.py`),
- the configuration script - `run_self_refine.sh`,
- the notebook used to generate the plots and do the analysis from the json files (`analysis_plotting.ipynb`), [if needed to run generate the json in results folder and run for each configuration]
- and the generated results (JSON files and figures) in the ./results directory.
- the `requirements.txt` file has been provided to reproduce the environment

## Configuration 1
- draft_temperature = 0.75
- feedback_temperature = 0.4
- refine_temperature = 0.1

## Configuration 2
- draft_temperature = 0.1
- feedback_temperature = 0.1
- refine_temperature = 0.1


## Environment Setup

To reproduce the experiments, first create and activate a new Python environment:

```bash
# Create a new virtual environment (recommended: conda or venv)
conda create -n self_refine_env python=3.10 -y
conda activate self_refine_env

# Install dependencies
pip install -r requirements.txt
```

## Run the Self Refine Pipeline using Bash

```bash
chmod +x run_self_refine.sh

./run_self_refine.sh
```