# Environment Setup

This guide explains how to recreate the `sam3_venv` conda environment used to run SAM3-FSVOS experiments.

## Prerequisites

- **Miniconda or Anaconda** installed
- **CUDA 12.6** available on the system (e.g. via `module load cuda/12.6` on HPC clusters)
- **Git** to clone the repository

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/TrueMaicol/sam3_fsvos
cd sam3_fsvos
```

## Step 2 — Create the conda environment

The environment uses Python 3.12 from the conda-forge channel and is created as a local prefix (inside the repo root):

```bash
conda create --prefix ./sam3_venv python=3.12 -c conda-forge
conda activate ./sam3_venv
```

## Step 3 — Install PyTorch (CUDA 12.6)

Install the PyTorch build matching **CUDA 12.6**. If your cluster uses a different CUDA version, replace `cu126` with the appropriate tag (e.g. `cu118`, `cu124`):

```bash
pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 torchaudio==2.7.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126
```

> The `nvidia-*` CUDA libraries (cublas, cudnn, etc.) are bundled inside the PyTorch wheel — do not install them separately.

## Step 4 — Install the SAM3 package in editable mode

From inside the `src/` directory:

```bash
cd src
pip install -e .
```

This installs `sam3` as an editable package so any changes to the source code are immediately reflected without reinstalling.

## Step 5 — Install the remaining dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` contains the full frozen environment (pinned versions) used for experiments, excluding PyTorch, SAM3, and the bundled NVIDIA libraries.

## Step 6 — Verify the installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import sam3; print('sam3 ok')"
```

---

## SLURM / HPC usage

On Leonardo (and similar HPC clusters), job scripts should load the CUDA module and activate the env before running:

```bash
module load cuda/12.6

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate ./sam3_venv
```

See the `SLURM_SCRIPTS/` directory for ready-to-use job submission scripts.
