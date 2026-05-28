#!/bin/bash
#SBATCH --job-name=persfl_fedprox_mu
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

# Sweep: 4 clusters x 4 mu values = 16 array tasks
N_CLUSTERS_LIST=(1 4 8 24)
MU_LIST=(0 0.01 0.1 1.0)

IDX=$SLURM_ARRAY_TASK_ID
SEED=$IDX

NUM_MU=${#MU_LIST[@]}

CLUSTER_IDX=$((IDX / NUM_MU))
MU_IDX=$((IDX % NUM_MU))

N_CLUSTERS=${N_CLUSTERS_LIST[$CLUSTER_IDX]}
MU=${MU_LIST[$MU_IDX]}

export PYTHONPATH=$PYTHONPATH:$PWD

OUT_DIR="results/cnn_cifar10_fedprox_mu"
mkdir -p "${OUT_DIR}/FedProx"

echo "Running FedProx | clusters=$N_CLUSTERS | mu=$MU | seed=$SEED"

srun python scripts/main.py \
    --n_clients 24 \
    --n_clusters "$N_CLUSTERS" \
    --n_samples 500 \
    --n_samples_test 1000 \
    --model cnn \
    --dataset cifar10 \
    --R 30 \
    --lrate 0.03 \
    --lrate_decay 0.99 \
    --algo FedProx \
    --R_local 10 \
    --mu "$MU" \
    --fname "${OUT_DIR}/FedProx/cnn_cifar10_c${N_CLUSTERS}_mu${MU}_seed${SEED}.csv" \
    --device cuda \
    --problem classification \
    --seed $SEED
