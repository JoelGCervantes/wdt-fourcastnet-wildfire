# 🔥 WDT FourCastNet — Wildfire Spread Forecasting

> Adapting the state-of-the-art FourCastNet weather model for wildfire spread forecasting as part of NASA's Wildfire Digital Twin (WDT) initiative. Includes multi-GPU training backbone, SLURM/HPC workflows, DDP optimization, and inference utilities.

---

## Overview

This repository contains the engineering infrastructure for training and running **FourCastNetv1** on large-scale atmospheric datasets (ERA5, 6TB+) with a focus on wildfire-relevant atmospheric variables. The work is part of a broader effort to build a **NASA Wildfire Digital Twin** — a coupled AI/physics system integrating FourCastNet with WRF-SFIRE to model emergent fire-atmosphere feedback dynamics.

### Key Contributions

- **High-performance DDP training backbone** on ORCA HPC (5 nodes × 4 GPUs, 40 CPUs/task) using `torchrun`
- **Resolved DDP synchronization bottlenecks** and optimized multi-worker data loading for 6TB+ ERA5 datasets
- **Multi-agent synchronization layer** between FourCastNet (neural operator) and WRF-SFIRE (physics solver) for bidirectional fire-atmosphere state updates
- **SLURM job submission workflows** tuned for multi-node PyTorch distributed training
- **Inference and evaluation utilities** for post-training analysis and visualization

---

## Repository Structure

```
wdt-fourcastnet-wildfire/
├── FCN/                    # FourCastNet model architecture & training code
├── visualizations/         # Evaluation plots and diagnostic outputs
├── inference.py            # Inference pipeline for trained checkpoints
├── submit_job.sh           # SLURM job submission script (ORCA HPC)
├── apex_download.sh        # NVIDIA Apex install helper for mixed-precision training
├── requirements.txt        # Python dependencies
└── .gitignore
```

---

## Setup

### Prerequisites

- Python ≥ 3.9
- PyTorch (with CUDA support)
- NVIDIA Apex (for mixed-precision training)
- Access to an HPC cluster with SLURM

### Installation

```bash
# Clone the repo
git clone https://github.com/JoelGCervantes/wdt-fourcastnet-wildfire.git
cd wdt-fourcastnet-wildfire

# Install Python dependencies
pip install -r requirements.txt

# Install NVIDIA Apex (required for optimized training)
bash apex_download.sh
```

---

## Training on HPC (SLURM / ORCA)

The training job is configured for **multi-node distributed training** using `torchrun` and PyTorch DDP.

### Job Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Nodes         | 5                |
| GPUs per node | 4 (20 total)     |
| CPUs per task | 40               |
| Dataset       | ERA5 (~6TB)      |
| Launcher      | `torchrun`       |
| Cluster       | ORCA HPC (SLURM) |

### Submitting a Training Job

```bash
sbatch submit_job.sh
```

Monitor your run via W&B (Weights & Biases) — the training backbone logs process memory, loss curves, and throughput metrics automatically.

---

## Inference

Run inference on a trained checkpoint:

```bash
python inference.py \
  --checkpoint /path/to/checkpoint.pt \
  --config /path/to/config.yaml \
  --output_dir ./visualizations
```

---

## Architecture Notes

### FourCastNet (FCN)

FourCastNetv1 is a vision transformer-based (ViT) neural operator trained on ERA5 global atmospheric reanalysis data. It uses **Adaptive Fourier Neural Operators (AFNO)** for efficient spectral-domain feature mixing. In this project, it is adapted to prioritize wildfire-relevant atmospheric variables (wind speed/direction, humidity, temperature).

### WRF-SFIRE Coupling (Multi-Agent Layer)

A synchronization layer treats FourCastNet and WRF-SFIRE as **independent agents** with bidirectional state-space updates:

```
FourCastNet (atmo state) ──→ WRF-SFIRE (fire spread)
        ↑                              │
        └──────── feedback loop ───────┘
```

This enables modeling of emergent **fire-atmosphere feedback dynamics** — where fire-generated heat modifies local atmospheric conditions, which in turn influence fire behavior (_ongoing_)

---

## Requirements

See [`requirements.txt`](requirements.txt) for the full list. Core dependencies:

- `torch`
- `torchvision`
- `numpy`
- `h5py`
- `wandb`
- `einops`
- `timm`

---

## Project Context

This work is part of the **NASA Wildfire Digital Twin** project at [wildfire-ai-psu.org](https://wdt.cecs.pdx.edu/), in collaboration with Portland State University. The broader goal is to build a real-time digital twin of wildfire behavior by coupling AI-based weather prediction with physics-based fire spread models.

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@misc{garciacervantes2025wdt,
  author       = {Joel Garcia-Cervantes},
  title        = {WDT FourCastNet: Wildfire Spread Forecasting on HPC},
  year         = {2025},
  organization = {NASA Wildfire Digital Twin / Penn State University},
  url          = {https://github.com/JoelGCervantes/wdt-fourcastnet-wildfire}
}
```

---

## License

This project is for research purposes. Please contact the author before use in derivative works.
