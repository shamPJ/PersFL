#!/bin/bash
#SBATCH --job-name=persfl_topk
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --array=0-39
#SBATCH --output=logs/out_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err

module load mamba
source activate pytorch-env-cuda118

# ── MODE: nclusters (default) or sigma ───────────────────────────────────────
# Submit with:  sbatch --export=ALL,MODE=nclusters scripts/run_cifar10_topk.sh
#               sbatch --export=ALL,MODE=sigma      run_cifar10_topk.sh
MODE=${MODE:-nclusters}

K_LIST=(4 8)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
NUM_SEEDS=${#SEEDS[@]}

IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$((IDX % NUM_SEEDS))
SEED=${SEEDS[$SEED_IDX]}

export PYTHONPATH=$PYTHONPATH:$PWD

if [ "$MODE" = "nclusters" ]; then
    # ── n_clusters sweep: 4 nc x 10 seeds = 40 tasks (#SBATCH --array=0-39) ──
    N_CLUSTERS_LIST=(1 4 8 24)
    NUM_NC=${#N_CLUSTERS_LIST[@]}
    NC_IDX=$((IDX / NUM_SEEDS % NUM_NC))
    NC=${N_CLUSTERS_LIST[$NC_IDX]}

    OUT_DIR="results/cnn_cifar10_topk_nclusters"
    mkdir -p $OUT_DIR

    for K in "${K_LIST[@]}"; do
        echo "Running Algorithm1_TopK | K=$K | n_clusters=$NC | seed=$SEED"
        srun python scripts/main.py \
            --n_clients 24 \
            --n_clusters $NC \
            --n_samples 500 \
            --n_samples_test 1000 \
            --model cnn \
            --dataset cifar10 \
            --algo Algorithm1_TopK \
            --R 30 \
            --R_local 10 \
            --S 8 \
            --K $K \
            --weighting reward \
            --lrate 0.01 \
            --lrate_decay 0.99 \
            --fname "${OUT_DIR}/cnn_cifar10_K${K}_nc${NC}_seed${SEED}.csv" \
            --device cuda \
            --problem classification \
            --seed $SEED
    done

elif [ "$MODE" = "sigma" ]; then
    # ── sigma sweep: 4 sigma x 10 seeds = 40 tasks (#SBATCH --array=0-39) ────
    SIGMA_LIST=(15 30 45 60)
    NUM_SIGMA=${#SIGMA_LIST[@]}
    SIGMA_IDX=$((IDX / NUM_SEEDS % NUM_SIGMA))
    SIGMA=${SIGMA_LIST[$SIGMA_IDX]}

    OUT_DIR="results/cnn_cifar10_topk_sigma"
    mkdir -p $OUT_DIR

    for K in "${K_LIST[@]}"; do
        echo "Running Algorithm1_TopK | K=$K | sigma=$SIGMA | seed=$SEED"
        srun python scripts/main.py \
            --n_clients 24 \
            --n_clusters 4 \
            --n_samples 500 \
            --n_samples_test 1000 \
            --model cnn \
            --dataset cifar10 \
            --algo Algorithm1_TopK \
            --R 30 \
            --R_local 10 \
            --S 8 \
            --K $K \
            --weighting reward \
            --lrate 0.01 \
            --lrate_decay 0.99 \
            --sigma $SIGMA \
            --fname "${OUT_DIR}/cnn_cifar10_K${K}_sigma${SIGMA}_seed${SEED}.csv" \
            --device cuda \
            --problem classification \
            --seed $SEED
    done

else
    echo "ERROR: unknown MODE=$MODE. Use MODE=nclusters or MODE=sigma." >&2
    exit 1
fi

