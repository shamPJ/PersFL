#!/bin/bash

D_LIST=(2)
SEEDS=(0 1 2)

# Output directory
OUT_DIR="results/linear_syn_dm_algo2"
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
            --n_clients 100 \
            --n_clusters 2 \
            --n_features $D \
            --model linreg \
            --dataset synthetic \
            --algo Algorithm2 \
            --R 500 \
            --R_local 0 \
            --lrate 0.01 \
            --S 20 \
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