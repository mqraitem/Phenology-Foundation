#!/bin/bash
# Activate your environment

#$ -P ivc-ml
#$ -l gpus=1
#$ -pe omp 6
#$ -j y
#$ -l h_rt=48:00:00
#$ -l gpu_memory=48G

export PATH=/projectnb/ivc-ml/mqraitem/miniconda3/bin:$PATH
source activate geo

# Run scaled patch experiment
# d_model scales quadratically with patch area to maintain consistent compression ratio
# patch_size=16: d_model=128, patch_size=32: d_model=512, patch_size=64: d_model=2048
python train_lsp_patch_scaled.py $args
