#!/bin/bash
# Script: train.sh
# Purpose: FourCastNet multi-node distributed training
# Author: garciac2

#SBATCH -p short
#SBATCH -t 4:00:00        # Training takes longer than inference
#SBATCH -J FCN_TRAIN
#SBATCH --output=logs/%x_%j.out
#SBATCH -N 5               # Number of nodes
#SBATCH --gres=gpu:4       # Use 1 GPU (change to gpu:2 if you want to speed up)
#SBATCH --ntasks-per-node=1	   # Launch only 1 manager task (torchrun handles multi-process spawning)
#SBATCH --cpus-per-task=40 # CPU cores (processors) assigned to each individual task (data loading) ***scale with number of gpus***
#SBATCH --mem=0

module load python/3.12.12
module load cuda/12.9.0

unset LD_LIBRARY_PATH

cd /scratch/wdt/ai/wdt-fcn-wildfire/FCN
mkdir -p logs

source ../env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID # calculates rank of node

# Use torchrun to launch training. 
# --nproc_per_node should match the number of GPUs requested in --gres.
srun torchrun \
    --nnodes=5 \
    --nproc_per_node=4 \
    --node_rank=$NODE_RANK \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    train.py \
    --config afno_backbone \
    --run_num 2
