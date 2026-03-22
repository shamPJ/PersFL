#!/bin/bash

# Different batch sizes to try
BATCH_SIZES=(10 20 30 50)

# Random seeds
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# Output directory
OUT_DIR="results/linear_syn_S"
mkdir -p $OUT_DIR

# Timing
START=$SECONDS
echo "Starting experiments at $(date)"

# Loop over batch sizes and seeds
for S in "${BATCH_SIZES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "========================================"
        echo "Running experiment: S=$S, seed=$SEED"
        echo "========================================"

        # Construct CSV file path
        F_PATH="${OUT_DIR}/linear_syn_S_${S}_${SEED}.csv"
        START_EXP=$SECONDS

        python main.py \
            --n_clients 150 \
            --n_clusters 3 \
            --n_features 10 \
            --model linreg \
            --dataset synthetic \
            --algo persfl \
            --R 500 \
            --lrate 0.01 \
            --S $S \
            --fname $F_PATH \
            --device cpu \
            --problem regression \
            --seed $SEED

        END_EXP=$SECONDS
        ELAPSED_EXP=$(( END_EXP - START_EXP ))
        echo "Experiment completed in $ELAPSED_EXP seconds"
    done
done

END=$SECONDS
ELAPSED=$(( END - START ))
echo "All experiments completed in $ELAPSED seconds"