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

exp="Algorithm1"

case "$exp" in
    Algorithm1)
        ARGS="--algo Algorithm1 --S 10 --R_local 5"
        LRATES=(0.01 0.02 0.03)
        SUBDIR="Algorithm1"
        ;;

    FedAvg)
        ARGS="--algo FedAvg --S 23 --R_local 10"
        LRATES=(0.05 0.06 0.07)
        SUBDIR="FedAvg"
        ;;
    
    FedProx)
        ARGS="--algo FedProx --S 23 --R_local 10 --mu 0.01"
        LRATES=(0.05 0.06 0.07)
        SUBDIR="FedProx"
        ;;

    *)
        echo "Unknown experiment: $exp"
        exit 1
        ;;
esac

# ===============================
# Sweep grids
# ===============================
N_CLUSTERS_LIST=(1 2 3 4)

IDX=$SLURM_ARRAY_TASK_ID

NUM_LR=${#LRATES[@]}

CLUSTER_IDX=$((IDX / NUM_LR))
LR_IDX=$((IDX % NUM_LR))

N_CLUSTERS=${N_CLUSTERS_LIST[$CLUSTER_IDX]}
LRATE=${LRATES[$LR_IDX]}

SEED=$IDX

export PYTHONPATH=$PYTHONPATH:$PWD

# Output directory
OUT_DIR="results/cnn_cifar10_iid_lrate"
mkdir -p $OUT_DIR

# Algorithm subdirectories
ALGOS=("Algorithm1" "Algorithm2" "FedAvg" "FedProx")
for ALG in "${ALGOS[@]}"; do
    mkdir -p "${OUT_DIR}/${ALG}"
done

srun python scripts/main.py \
    --n_clients 24 \
    --n_clusters "$N_CLUSTERS" \
    --n_classes 10 \
    --n_samples 100 \
    --n_samples_val 1000 \
    --model cnn \
    --dataset cifar10 \
    --R 50 \
    --lrate $LRATE \
    --lrate_decay 0.98 \
    $ARGS \
    --fname "${OUT_DIR}/${SUBDIR}/cnn_cifar10_iid_c${N_CLUSTERS}_lr${LRATE}_seed${SEED}.csv" \
    --device cuda \
    --problem classification \
    --seed $SEED 




