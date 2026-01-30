#!/bin/bash -l
#SBATCH -p short
#SBATCH -t 01:00:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH -J fcn_32gpu_train
#SBATCH -o logs/%x_%j.out


# Config Vars
config_file=./config/AFNO.yaml
config='afno_backbone'
run_num='1'

#export HDF5_USE_FILE_LOCKING=FALSE # Disables file locking to prevent crashes on shared network file systems

# Networking for Multi-Node Communication
# We get the first node's name to act as the "Master" node
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) # Using scontrol ensures all nodes talk to the ACTUAL hostname of the first node
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 4)) # 8 nodes * 4 GPUs = 32

# Debug Info 
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

source env/bin/activate # activate env

echo "MASTER_ADDR=$MASTER_ADDR"
echo "TOTAL_GPUS=$WORLD_SIZE"

# Launch with srun + torchrun
# We use srun to start 1 torchrun process on each of the 8 nodes.
# Each torchrun will then manage the 4 GPUs on its specific node.
srun -u bash -c "unset LD_LIBRARY_PATH && torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train2.py \
    --enable_amp \
    --yaml_config=$config_file \
    --config=$config \
    --run_num=$run_num"