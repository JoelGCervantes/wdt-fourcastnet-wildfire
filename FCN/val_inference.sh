#!/bin/bash
# Script: val_inference.sh
# Purpose: Historical Validation (Testing model accuracy across different dates using RMSE and ACC)
# Author: garciac2

#SBATCH -p normal
#SBATCH -t 06:00:00
#SBATCH -J FCN_HIST_VAL
#SBATCH --output=logs/%x_%j.out
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=0

module load python/3.12.9
module load cuda/12.9.0

unset LD_LIBRARY_PATH # Resolves CUDA/PyTorch version conflicts

cd /scratch/wdt/ai/wdt-fcn-wildfire/FCN

mkdir -p logs

source ../env/bin/activate

export PYTHONPATH=$PYTHONPATH:$(pwd)

# to limit number of initial condtions use: --n_initial_conditions 
python -m inference.inference --config afno_backbone --run_num 0


