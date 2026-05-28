#!/bin/bash
#SBATCH --job-name=persfl_dm
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=3 # task is job instance created from the array; each task runs .sh independently
#SBATCH --gres=gpu:1
#SBATCH --array=0-39
#SBATCH --output=logs/out_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err

# ===============================
# Environment
# ===============================
module load mamba
source activate pytorch-env-cuda118

N_CLUSTERS_LIST=(1 4 8 24)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

IDX=$SLURM_ARRAY_TASK_ID

export PYTHONPATH=$PYTHONPATH:$PWD

# Output directory
OUT_DIR="results/cnn_cifar10_rotated"
mkdir -p $OUT_DIR

# ===============================
# Algorithms
# ===============================
ALGOS=("Algorithm1" "FedAvg" "FedBN" "FedProx" "IFCA" "Ditto")

for ALG in "${ALGOS[@]}"; do

    case "$ALG" in
        Algorithm1)
            ARGS="--algo Algorithm1 --S 8 --R_local 10 --lrate 0.01"
            ;;
        
        Algorithm2)
            ARGS="--algo Algorithm2 --S 8 --R_local 10 --lrate 0.01"
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
            LRATES=(0.01 0.02 0.03)
            ;;
    esac

    NS=${#SEEDS[@]}

    CLUSTER_IDX=$((IDX / NS))
    SEED_IDX=$((IDX % NS))

    N_CLUSTERS=${N_CLUSTERS_LIST[$CLUSTER_IDX]}
    SEED=${SEEDS[$SEED_IDX]}

    mkdir -p "${OUT_DIR}/${ALG}"

    echo "Running $ALG | clusters=$N_CLUSTERS | seed=$SEED"
    srun python scripts/main.py \
        --n_clients 24 \
        --n_clusters "$N_CLUSTERS" \
        --n_samples 500 \
        --n_samples_test 1000 \
        --model cnn \
        --dataset cifar10 \
        --R 30 \
        --lrate_decay 0.99 \
        $ARGS \
        --fname "${OUT_DIR}/${ALG}/cnn_cifar10_c${N_CLUSTERS}_seed${SEED}.csv" \
        --device cuda \
        --problem classification \
        --seed $SEED


    if [ "$ALG" == "IFCA" ] && [ "$N_CLUSTERS" -gt 1 ]; then

        ALGO_K=$((N_CLUSTERS / 2))
        OUT_DIR_K="results/cnn_cifar10_ifca_misspec/IFCA_k${ALGO_K}"
        mkdir -p "$OUT_DIR_K"

        echo "Running IFCA misspec | true_clusters=$N_CLUSTERS | algo_K=$ALGO_K | seed=$SEED"
        srun python scripts/main.py \
            --n_clients 24 \
            --n_clusters "$N_CLUSTERS" \
            --algo_n_clusters "$ALGO_K" \
            --n_samples 500 \
            --n_samples_test 1000 \
            --model cnn \
            --dataset cifar10 \
            --R 30 \
            --lrate_decay 0.99 \
            --algo IFCA --R_local 10 --lrate 0.03 \
            --fname "${OUT_DIR_K}/cnn_cifar10_c${N_CLUSTERS}_seed${SEED}.csv" \
            --device cuda \
            --problem classification \
            --seed $SEED
    fi

done

