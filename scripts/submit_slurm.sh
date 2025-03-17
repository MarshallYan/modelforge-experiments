#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=10:00:00
#SBATCH --output=%j_%x_%N.out
#SBATCH --error=%j_%x_%N.err

source ${HOME}/.bashrc

# Report node in use
hostname

# Disable NCC
export NCCL_P2P_DISABLE=1

# Report CUDA info
env | sort | grep 'CUDA'

# Report GPUs available
nvidia-smi

# Activate environment
mamba activate mf-train

# Execute the python command
pwd
srun python perform_training.py --condensed_config_path config.toml --accelerator 'gpu' --device [0]