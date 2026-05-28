#!/bin/bash
#SBATCH --job-name=persfl_rlocal
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --array=0-15
#SBATCH --output=logs/out_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err

# ===============================
# Environment
# ===============================
module load mamba
source activate pytorch-env-cuda118

# Sweep: 4 clusters x 4 S values = 16 array tasks
N_CLUSTERS_LIST=(1 2 3 4)
S_LIST=(4 8 12 16)

IDX=$SLURM_ARRAY_TASK_ID
SEED=$IDX

NUM_S=${#S_LIST[@]}

CLUSTER_IDX=$((IDX / NUM_S))
S_IDX=$((IDX % NUM_S))

N_CLUSTERS=${N_CLUSTERS_LIST[$CLUSTER_IDX]}
S=${S_LIST[$S_IDX]}

export PYTHONPATH=$PYTHONPATH:$PWD

OUT_DIR="results/cnn_cifar10_S"
mkdir -p "${OUT_DIR}/Algorithm1"

echo "Running Algorithm1 | clusters=$N_CLUSTERS | S=$S | seed=$SEED"

srun python scripts/main.py \
    --n_clients 24 \
    --n_clusters "$N_CLUSTERS" \
    --n_classes 10 \
    --n_samples 500 \
    --n_samples_test 1000 \
    --model cnn \
    --dataset cifar10 \
    --R 30 \
    --R_local 10 \
    --lrate 0.01 \
    --lrate_decay 0.99 \
    --algo Algorithm1 \
    --S "$S" \
    --fname "${OUT_DIR}/Algorithm1/cnn_cifar10_c${N_CLUSTERS}_S${S}_seed${SEED}.csv" \
    --device cuda \
    --problem classification \
    --seed $SEED

