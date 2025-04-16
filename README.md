# LDM_res: Latent Diffusion Model for Meteorological Downscaling

<div align="center">
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
  <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
  <a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
  <a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
</div>

## Overview

**LDM_res** implements a physics‑conditioned Latent Diffusion Model (LDM) for statistical downscaling of meteorological fields.  Given coarse ERA5 reanalysis inputs and optional static high‑resolution features, LDM_res reconstructs fine‑scale 2 m temperature (and UV wind components) with improved physical consistency via a novel PDE‑based regularization.

This repository provides:
- Training and inference code for the LDM_res model and baselines.
- Utilities to download sample/full datasets and pretrained checkpoints.
- Configuration templates for distributed training and single‑GPU workflows.
- Notebooks for evaluation metrics and result visualization, including physical‑loss computation.

## Acknowledgments

This work builds heavily on the [DiffScaler](https://github.com/DSIP-FBK/DiffScaler) codebase.  We extend and adapt implementations by **Elena Tomasi**, **Gabriele Franch**, and **Marco Cristoforetti**—many thanks for their foundational contributions.

## Installation

```bash
# Clone the project
git clone https://github.com/YourGithubName/ldm_res.git
cd ldm_res

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Data & Model Download

### Sample Dataset (~45 GB)
```bash
cd data
bash download_sample_dataset.sh
```

### Full Dataset (~330 GB)
```bash
cd data
bash download_full_dataset.sh
# Alternative stable download if issues arise:
bash Better_download_full_dataset.sh
```

### Pretrained Checkpoints (~12 GB)
```bash
cd pretrained_models
bash download_pretrained_models.sh
```

## Configuration

- **Parallel training:** YAML specs in `configs/trainer/` enable multi‑node/GPU jobs via PyTorch Lightning.
- **Experiment configs:** see `configs/experiment/downscaling_LDM_res_2mT.yaml`, etc.
  - Define dataset paths, model hyperparameters, logging, and checkpointing.
  - **NEW** fields:
    ```yaml
    trainable_parts:
      - "denoiser.output_blocks"
      - "autoencoder.decoder"
      - "denoiser.middle_block"
    ```
    to restrict training to submodules (compute‑limited mode).
  - All PDE‑loss weightings (`lambda_PDE`, etc.) are specified in these files.

## Training

### Full‑scale LDM (100 GB VRAM required)
```bash
python3 src/train.py experiment=downscaling_LDM_res_2mT
```
- Requires GPUs with ≥ 100 GB VRAM for end‑to‑end LDM fine‑tuning.
- Leverages distributed/multi‑GPU parallelism via `configs/trainer/*.yaml`.

### Compute‑Limited Mode (Single‑GPU)
- Edit `configs/experiment/downscaling_LDM_res_2mT.yaml`:
  - Set `trainable_parts` as shown above.
- Launch on a single GPU with:
```bash
python3 src/train.py experiment=downscaling_LDM_res_2mT
```
- Note: performance may degrade when training only decoder and selected blocks.

### SLURM Submission
- A template submission script is provided at `configs/experiment/Submitscript.sh`.
- **Edit file paths** to match your cluster environment before use.

> **Important:** We do **not** modify the UV predictor, UNET, or GAN architectures—only LDM components.

## Inference & Evaluation

- **GPU requirement:** ≥ 16 GB VRAM for single‑frame inference.
- Notebooks in `notebooks/` guide:
  - `models_inference.ipynb` for loading pretrained models and running inference.
  - `Fig_snapshots.ipynb` and others to reproduce visualizations and metrics.
- Metric computation and inference code has been extended to incorporate the PDE‑based physical loss—see `src/models/ldm_module.py` for implementation details.


## Further Resources

For detailed usage patterns, advanced configurations, and troubleshooting tips, please refer to the original DiffScaler README and documentation.

