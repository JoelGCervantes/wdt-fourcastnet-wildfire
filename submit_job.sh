#!/bin//bash
#SBATCH -p normal
#SBATCH -t 06:00:00
#SBATCH -J CIFAR10_CNN
#SBATCH --output=logs/%x_%j.out
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=28G

module load python/3.12.9
module load cuda/12.9.0

mkdir -p logs

source env/bin/activate

python inference.py



