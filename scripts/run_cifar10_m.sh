#!/bin/bash
#SBATCH --job-name=persfl_cifar10_m
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

# Sweep: 4 clusters x 4 M values = 16 array tasks (algos looped inside)
N_CLUSTERS_LIST=(1 4 8 24)
M_LIST=(50 100 200 500)

IDX=$SLURM_ARRAY_TASK_ID

NUM_M=${#M_LIST[@]}

CLUSTER_IDX=$((IDX / NUM_M))
M_IDX=$((IDX % NUM_M))

SEED=$IDX

N_CLUSTERS=${N_CLUSTERS_LIST[$CLUSTER_IDX]}
M=${M_LIST[$M_IDX]}

export PYTHONPATH=$PYTHONPATH:$PWD

OUT_DIR="results/cnn_cifar10_m"

echo "========================================"
echo "  clusters = $N_CLUSTERS"
echo "  M        = $M"
echo "  seed     = $SEED"
echo "========================================"

# ===============================
# Algorithms
# ===============================
ALGOS=("Algorithm1" "FedAvg" "FedBN" "FedProx" "IFCA" "Ditto" "Algorithm1_TopK_K4" "Algorithm1_TopK_K8")

for ALG in "${ALGOS[@]}"; do

    case "$ALG" in
        Algorithm1)
            ARGS="--algo Algorithm1 --S 8 --R_local 10 --lrate 0.01"
            ;;
        FedAvg)
            ARGS="--algo FedAvg --R_local 10 --lrate 0.03"
            ;;
        FedBN)
            ARGS="--algo FedBN --R_local 10 --lrate 0.03"
            ;;
        FedProx)
            ARGS="--algo FedProx --mu 0.01 --R_local 10 --lrate 0.03"
            ;;
        IFCA)
            ARGS="--algo IFCA --R_local 10 --lrate 0.03"
            ;;
        Ditto)
            ARGS="--algo Ditto --lmbd 0.1 --R_local 10 --lrate 0.03"
            ;;
        Algorithm1_TopK_K4)
            ARGS="--algo Algorithm1_TopK --K 4 --S 8 --R_local 10 --lrate 0.01"
            ;;
        Algorithm1_TopK_K8)
            ARGS="--algo Algorithm1_TopK --K 8 --S 8 --R_local 10 --lrate 0.01"
            ;;
    esac

    mkdir -p "${OUT_DIR}/${ALG}"
    echo "Running $ALG | clusters=$N_CLUSTERS | m=$M | seed=$SEED"

    srun python scripts/main.py \
        --n_clients 24 \
        --n_clusters "$N_CLUSTERS" \
        --n_classes 10 \
        --n_samples "$M" \
        --n_samples_test 1000 \
        --model cnn \
        --dataset cifar10 \
        --R 30 \
        --lrate_decay 0.99 \
        $ARGS \
        --fname "${OUT_DIR}/${ALG}/cnn_cifar10_c${N_CLUSTERS}_m${M}_seed${SEED}.csv" \
        --device cuda \
        --problem classification \
        --seed $SEED

done

