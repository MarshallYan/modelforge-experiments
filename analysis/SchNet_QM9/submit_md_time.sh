#!/bin/bash
#SBATCH --job-name=schnet_qm9_md_time
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=20:00:00
#SBATCH --output=slurm_out/%j_%x_%N.out
#SBATCH --error=slurm_err/%j_%x_%N.err

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
mamba activate modelforge

# Execute the python command
pwd
python md_timer.py
