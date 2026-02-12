#!/bin/bash
# Script: train.sh
# Purpose: Training FourCastNet on Wildfire data
# Author: garciac2

#SBATCH -p normal
#SBATCH -t 24:00:00        # Training takes longer than inference
#SBATCH -J FCN_TRAIN
#SBATCH --output=logs/%x_%j.out
#SBATCH -N 10               # Number of nodes
#SBATCH --gres=gpu:4       # Use 1 GPU (change to gpu:2 if you want to speed up)
#SBATCH --ntasks=1	   # Launch only 1 manager task (torchrun handles multi-process spawning)
#SBATCH --cpus-per-task=20 # CPU cores (processors) assigned to each individual task (data loading) ***scale with number of gpus***
#SBATCH --mem=0

module load python/3.12.9
module load cuda/12.9.0

unset LD_LIBRARY_PATH

cd /scratch/wdt/ai/wdt-fcn-wildfire/FCN
mkdir -p logs

source ../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Use torchrun to launch training. 
# --nproc_per_node should match the number of GPUs requested in --gres.
torchrun --nproc_per_node=2 \
    train.py \
    --config afno_backbone \
    --run_num 1 \
