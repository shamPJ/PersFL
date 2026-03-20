#!/bin/bash

# Different numbers of clusters to try
CLUSTERS=(2 3 5 10)

# Random seeds
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# Output directory
OUT_DIR="results/linear_syn_nclusters"
mkdir -p $OUT_DIR

# Timing
START=$SECONDS
echo "Starting experiments at $(date)"

# Loop over clusters and seeds
for N_CLUSTERS in "${CLUSTERS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "========================================"
        echo "Running experiment: n_clusters=$N_CLUSTERS, seed=$SEED"
        echo "========================================"

        # Construct CSV file path
        F_PATH="${OUT_DIR}/linear_syn_nclusters_${N_CLUSTERS}_${SEED}.csv"
        START_EXP=$SECONDS

        python main.py \
            --n_clients 150 \
            --n_clusters $N_CLUSTERS \
            --n_features 10 \
            --model linreg \
            --dataset synthetic \
            --algo persfl \
            --R 1500 \
            --lrate 0.01 \
            --S 30 \
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