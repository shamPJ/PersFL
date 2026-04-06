#!/bin/bash

D_LIST=(2)
SEEDS=(0 1 2)

# Output directory
export PYTHONPATH=$PYTHONPATH:$PWD

OUT_DIR="results/linear_syn_dm"
mkdir -p $OUT_DIR

# Algorithm subdirectories
ALGOS=("Algorithm2_SKLearn")
for ALG in "${ALGOS[@]}"; do
    mkdir -p "${OUT_DIR}/${ALG}"
done

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
        START_EXP=$SECONDS

        python scripts/main.py \
            --n_clients 100 \
            --n_clusters 3 \
            --n_features $D \
            --model decision_tree \
            --lmbd 0.05 \
            --dataset synthetic \
            --algo Algorithm2_SKLearn \
            --R 200 \
            --S 30 \
            --fname ${OUT_DIR}/Algorithm2_SKLearn/dt_synthetic_${SEED}.csv \
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