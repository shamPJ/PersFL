#!/bin/bash
#SBATCH --job-name=ditto_lambda
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

N_CLUSTERS_LIST=(1 4 8 24)
M_LIST=(50 100 200 500)
LMBD_LIST=(0.01 0.1 1.0 10.0)

IDX=$SLURM_ARRAY_TASK_ID
SEED=$IDX

NUM_M=${#M_LIST[@]}

CLUSTER_IDX=$((IDX / NUM_M))
M_IDX=$((IDX % NUM_M))

N_CLUSTERS=${N_CLUSTERS_LIST[$CLUSTER_IDX]}
N_SAMPLES=${M_LIST[$M_IDX]}

export PYTHONPATH=$PYTHONPATH:$PWD

OUT_DIR="results/cnn_cifar10_ditto"
mkdir -p "${OUT_DIR}"

for LMBD in "${LMBD_LIST[@]}"; do
    echo "Running Ditto | clusters=$N_CLUSTERS | n_samples=$N_SAMPLES | lmbd=$LMBD | seed=$SEED"

    srun python scripts/main.py \
        --n_clients 24 \
        --n_clusters "$N_CLUSTERS" \
        --n_classes 10 \
        --n_samples "$N_SAMPLES" \
        --n_samples_test 1000 \
        --model cnn \
        --dataset cifar10 \
        --R 30 \
        --R_local 10 \
        --lrate 0.01 \
        --lrate_decay 0.99 \
        --algo Ditto \
        --lmbd "$LMBD" \
        --fname "${OUT_DIR}/cnn_cifar10_c${N_CLUSTERS}_m${N_SAMPLES}_lmbd${LMBD}_seed${SEED}.csv" \
        --device cuda \
        --problem classification \
        --seed $SEED
done
