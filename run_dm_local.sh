#!/bin/bash

# D_LIST=(2 10 50 100)
D_LIST=(2 10)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# Output directory
OUT_DIR="results/linear_syn_dm"
mkdir -p $OUT_DIR

# timing 
START=$SECONDS
echo "Starting experiments at $(date)"

# Loop over all combinations
for D in "${D_LIST[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "========================================"
        echo "Running experiment: D=$D, seed=$SEED"
        echo "========================================"

        # Construct CSV file path
        F_PATH="${OUT_DIR}/linear_syn_dm_${D}_${SEED}.csv"
        START_EXP=$SECONDS

        python main.py \
            --n_features $D \
            --seed $SEED \
            --n_clients 100 \
            --model linreg \
            --dataset synthetic \
            --algo persfl \
            --R 500 \
            --lrate 0.01 \
            --S 20 \
            --fname $F_PATH \
            --device cpu
        END_EXP=$SECONDS
        ELAPSED_EXP=$(( END_EXP - START_EXP ))
        echo "Experiment completed in $ELAPSED_EXP seconds"
    done
done
END=$SECONDS
ELAPSED=$(( END - START ))
echo "All experiments completed in $ELAPSED seconds"