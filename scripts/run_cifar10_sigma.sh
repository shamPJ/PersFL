#!/bin/bash
#SBATCH --job-name=persfl_sigma
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --array=0-49
#SBATCH --output=logs/out_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err

# ===============================
# Environment
# ===============================
module load mamba
source activate pytorch-env-cuda118

SIGMA_LIST=(0 15 30 45 60)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

NS=${#SEEDS[@]}   # 10
IDX=$SLURM_ARRAY_TASK_ID

SIGMA_IDX=$((IDX / NS))
SEED_IDX=$((IDX % NS))

SIGMA=${SIGMA_LIST[$SIGMA_IDX]}
SEED=${SEEDS[$SEED_IDX]}

export PYTHONPATH=$PYTHONPATH:$PWD

# Output directory
OUT_DIR="results/cnn_cifar10_sigma"
mkdir -p $OUT_DIR

# ===============================
# Algorithms
# ===============================
#ALGOS=("Algorithm1" "FedAvg" "FedBN" "FedProx" "IFCA")
ALGOS=("Ditto")

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

    esac

    mkdir -p "${OUT_DIR}/${ALG}"

    echo "Running $ALG | sigma=$SIGMA | seed=$SEED"
    srun python scripts/main.py \
        --n_clients 24 \
        --n_clusters 4 \
        --n_samples 500 \
        --n_samples_test 1000 \
        --model cnn \
        --dataset cifar10 \
        --R 30 \
        --lrate_decay 0.99 \
        --sigma $SIGMA \
        $ARGS \
        --fname "${OUT_DIR}/${ALG}/cnn_cifar10_sigma${SIGMA}_seed${SEED}.csv" \
        --device cuda \
        --problem classification \
        --seed $SEED

done

