#!/bin/bash
#SBATCH --job-name=${JOB_NAME:-persfl_exp}
#SBATCH --time=${TIME:-04:00:00}
#SBATCH --mem=${MEM:-16G}
#SBATCH --cpus-per-task=${CPUS:-3}
#SBATCH --gres=gpu:${GPUS:-1}
#SBATCH --array=${ARRAY:-0-39}
#SBATCH --output=logs/out_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err

module load mamba
source activate pytorch-env

# Load config file
CONFIG_FILE="configs/experiment.yaml"

# Run Python script with SLURM array ID
srun python scripts/run_with_config.py \
    --config $CONFIG_FILE \
    --slurm_array_id $SLURM_ARRAY_TASK_ID