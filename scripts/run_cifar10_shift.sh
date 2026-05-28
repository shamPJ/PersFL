#!/bin/bash
#SBATCH --job-name=cifar10_shift
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=logs/out_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err

module load mamba
source activate pytorch-env-cuda118

#ALGOS=("Algorithm1" "Algorithm1_TopK" "FedAvg" "FedBN" "FedProx" "IFCA" "Ditto")
ALGOS=("Algorithm1_TopK")
SHIFT_AT_LIST=(5 10 15 20)

NUM_SHIFT=${#SHIFT_AT_LIST[@]}

IDX=$SLURM_ARRAY_TASK_ID
SEED=$IDX
ALGO=${ALGOS[$((IDX / NUM_SHIFT))]}
SHIFT_AT=${SHIFT_AT_LIST[$((IDX % NUM_SHIFT))]}

case "$ALGO" in
    Algorithm1)
        ALGO_ARGS="--S 8 --lrate 0.01"
        ;;
    Algorithm1_TopK)
        ALGO_ARGS="--K 4 --S 8 --lrate 0.01"
        ;;
    FedAvg)
        ALGO_ARGS="--lrate 0.03"
        ;;
    FedBN)
        ALGO_ARGS="--lrate 0.03"
        ;;
    FedProx)
        ALGO_ARGS="--mu 0.01 --lrate 0.03"
        ;;
    IFCA)
        ALGO_ARGS="--lrate 0.03"
        ;;
    Ditto)
        ALGO_ARGS="--lmbd 0.1 --lrate 0.03"
        ;;
esac

export PYTHONPATH=$PYTHONPATH:$PWD

OUT_DIR="results/cnn_cifar10_shift"
mkdir -p "${OUT_DIR}/${ALGO}"

echo "Running $ALGO | shift_at=$SHIFT_AT | seed=$SEED"

srun python scripts/main.py \
    --algo "$ALGO" \
    --dataset cifar10_shifted \
    --model cnn \
    --problem classification \
    --n_clients 24 \
    --n_clusters 4 \
    --n_classes 10 \
    --n_samples 500 \
    --n_samples_test 1000 \
    --R 30 \
    --R_local 10 \
    --lrate_decay 0.99 \
    --shift_at "$SHIFT_AT" \
    --seed "$SEED" \
    --fname "${OUT_DIR}/${ALGO}/cnn_cifar10_shift${SHIFT_AT}_seed${SEED}.csv" \
    --device cuda \
    $ALGO_ARGS
