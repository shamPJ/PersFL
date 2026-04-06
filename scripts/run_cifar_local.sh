#!/bin/bash
SEEDS=(0)

# Output directory
export PYTHONPATH=$PYTHONPATH:$PWD

OUT_DIR="results/cnn_cifar10_iid"
mkdir -p $OUT_DIR
# mkdir -p $OUT_DIR/Algorithm1
mkdir -p $OUT_DIR/FedAvg

# timing 
START=$SECONDS
echo "Starting experiments at $(date)"

# Loop over all combinations
for SEED in "${SEEDS[@]}"; do
    echo "========================================"
    echo "Running experiment: seed=$SEED"
    echo "========================================"

    # Construct CSV file path
    F_PATH="${OUT_DIR}/results/cnn_cifar10_iid_${SEED}.csv"
    START_EXP=$SECONDS

    python scripts/main.py \
        --n_clients 50 \
        --n_clusters 1 \
        --n_classes 10 \
        --model cnn \
        --dataset cifar10 \
        --algo FedAvg \
        --R 200 \
        --R_local 5 \
        --lrate 0.01 \
        --momentum 0.9 \
        --lrate_decay 0.999 \
        --S 30 \
        --fname ${OUT_DIR}/FedAvg/cnn_cifar10_iid_${SEED}.csv \
        --device cpu \
        --problem classification \
        --seed $SEED 

    END_EXP=$SECONDS
    ELAPSED_EXP=$(( END_EXP - START_EXP ))
    echo "Experiment completed in $ELAPSED_EXP seconds"
done
END=$SECONDS
ELAPSED=$(( END - START ))
echo "All experiments completed in $ELAPSED seconds"