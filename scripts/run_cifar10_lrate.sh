#!/bin/bash
#SBATCH --job-name=persfl_dm
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=3 # task is job instance created from the array; each task runs .sh independently
#SBATCH --gres=gpu:1
#SBATCH --array=0-11
#SBATCH --output=logs/out_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err

# ===============================
# Environment
# ===============================
module load mamba
source activate pytorch-env-cuda118

N_CLUSTERS_LIST=(1 4 8 24)
IDX=$SLURM_ARRAY_TASK_ID
SEED=$IDX

export PYTHONPATH=$PYTHONPATH:$PWD

# Output directory
OUT_DIR="results/cnn_cifar10_lrate"
mkdir -p $OUT_DIR

# ===============================
# Algorithms
# ===============================
#ALGOS=("Algorithm1" "FedBN" "FedProx" "IFCA")
ALGOS=("Ditto")

for ALG in "${ALGOS[@]}"; do

    case "$ALG" in
        Algorithm1)
            ARGS="--algo Algorithm1 --S 8 --R_local 10"
            LRATES=(0.01 0.02 0.03)
            ;;
        
        Algorithm2)
            ARGS="--algo Algorithm2 --S 8 --R_local 10"
            LRATES=(0.01 0.02 0.03)
            ;;
        
        FedBN)
            ARGS="--algo FedBN --R_local 10"
            LRATES=(0.01 0.02 0.03)
            ;;
        
        FedProx)
            ARGS="--algo FedProx --mu 0.01 --R_local 10"
            LRATES=(0.01 0.02 0.03)
            ;;
        
        IFCA)
            ARGS="--algo IFCA --R_local 10"
            LRATES=(0.02 0.03 0.04)
            ;;

        Ditto)
            ARGS="--algo Ditto --R_local 10"
            LRATES=(0.01 0.02 0.03)
            ;;
    esac

    NUM_LR=${#LRATES[@]}

    CLUSTER_IDX=$((IDX / NUM_LR))
    LR_IDX=$((IDX % NUM_LR))

    N_CLUSTERS=${N_CLUSTERS_LIST[$CLUSTER_IDX]}
    LRATE=${LRATES[$LR_IDX]}

    mkdir -p "${OUT_DIR}/${ALG}"

    echo "Running $ALG | clusters=$N_CLUSTERS | lr=$LRATE"

    srun python scripts/main.py \
        --n_clients 24 \
        --n_clusters "$N_CLUSTERS" \
        --n_samples 500 \
        --n_samples_test 1000 \
        --model cnn \
        --dataset cifar10 \
        --R 30 \
        --lrate $LRATE \
        --lrate_decay 0.99 \
        $ARGS \
        --fname "${OUT_DIR}/${ALG}/cnn_cifar10_c${N_CLUSTERS}_lr${LRATE}_seed${IDX}.csv" \
        --device cuda \
        --problem classification \
        --seed $IDX

done



