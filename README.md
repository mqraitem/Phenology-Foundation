# Phenology Prediction with Prithvi-EO

Fine-tuning geospatial foundation models for vegetation phenology estimation from satellite imagery.

## Overview

This repository predicts **Land Surface Phenology (LSP)** stages from multi-temporal satellite data:
- **Greenup** - Start of vegetation growth
- **Maturity** - Peak greenness
- **Senescence** - Start of decline
- **Dormancy** - Vegetation dormant

Predictions are day-of-year (DOY) values for each phenological stage.

## Data

- **Source**: Harmonized Landsat/Sentinel-2 (HLS) composites
- **Bands**: 6 spectral bands (B02, B03, B04, B05, B06, B07)
- **Temporal**: 12 time steps per sample
- **Spatial**: 330x330 pixel tiles
- **Labels**: Ground truth DOY for each phenology stage

Ground truth data: [NASA ORNL DAAC LSP Dataset](https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-landsat8-sentinel2-phenocam-2248-1)

Data paths are configured in `dirs.txt` and managed by `path_config.py`.

## Models

| Model | Description | Training Script |
|-------|-------------|-----------------|
| **Prithvi** | Full fine-tuning of Prithvi-EO-V2 | `train_prithvi.py` |
| **Prithvi-LoRA** | Parameter-efficient LoRA adaptation | `train_prithvi_lora.py` |
| **Prithvi-Mini** | Smaller Prithvi variant | `train_mini_prithvi.py` |
| **Temporal Transformer (Pixels)** | Lightweight pixel-level transformer | `train_lsp_pixels.py` |
| **Temporal Transformer (Patches)** | Patch-based transformer | `train_lsp_patch.py` |

## Usage

### Training Prithvi (Full Fine-tuning)
```bash
python train_prithvi.py --learning_rate 1e-4 --batch_size 2 --data_percentage 1.0
```

### Training Prithvi with LoRA
```bash
python train_prithvi_lora.py --learning_rate 1e-5 --lora_r 16 --lora_alpha 16 --batch_size 2
```

### Training Lightweight Transformer
```bash
python train_lsp_pixels.py --learning_rate 1e-4 --batch_size 8
```

### Key Arguments
- `--learning_rate`: Learning rate (default varies by model)
- `--batch_size`: Batch size
- `--data_percentage`: Fraction of training data to use (0.2, 0.4, 0.6, 0.8, 1.0)
- `--temporal_only`: Use temporal attention only (Prithvi models)
- `--lora_r`, `--lora_alpha`: LoRA rank and alpha (LoRA model)
- `--wandb`: Enable Weights & Biases logging

### Cluster Submission
Batch scripts are in `run_scripts/`. Example:
```bash
qsub run_scripts/train_prithvi_lora.sh
```

Or use `run_prithvi.py` to launch hyperparameter sweeps.

## Configuration

- `configs/prithvi_300m.yaml` - Prithvi-EO-V2 300M config
- `configs/prithvi_600m.yaml` - Prithvi-EO-V2 600M config
- `dirs.txt` - Data and checkpoint paths
