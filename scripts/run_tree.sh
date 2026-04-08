#!/bin/bash
#SBATCH --job-name=persfl_tree
#SBATCH --time=05:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3 # task is job instance created from the array; each task runs .sh independently
#SBATCH --array=0-39
#SBATCH --output=logs/out_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err

# ===============================
# Environment
# ===============================
module load mamba
source activate pytorch-env

export PYTHONPATH=$PYTHONPATH:$PWD
export OMP_NUM_THREADS=1

SEEDS=(0 1 2 3 4 5 6 7 8 9)

OUT_DIR="results"
mkdir -p $OUT_DIR

# subdirectories
EXPS=("linear_syn_dm" "linear_syn_S" "linear_syn_lmbd" "linear_syn_noise" "linear_syn_noise_w" "linear_syn_nclusters")
ALG="Algorithm2_SKLearn"
for EXP in "${EXPS[@]}"; do
    mkdir -p "${OUT_DIR}/${EXP}/${ALG}"
done

START=$SECONDS
echo "Starting experiments at $(date)"

# PARAM_LIST=(2 10 50 100)
# # ===============================
# # Index mapping
# # ===============================
# IDX=$SLURM_ARRAY_TASK_ID

# NP=${#PARAM_LIST[@]} # number of elements in the array
# NS=${#SEEDS[@]}

# # Total jobs = 4 * 10 = 40

# PARAM_IDX=$(( IDX / NS ))
# SEED_IDX=$(( IDX %  NS ))

# PARAM=${PARAM_LIST[$PARAM_IDX]}
# SEED=${SEEDS[$SEED_IDX]}

# echo "========================================"
# echo "Running experiment on GPU"
# echo "  D       = $PARAM"
# echo "  seed    = $SEED"
# echo "  GPU     = $CUDA_VISIBLE_DEVICES"
# echo "========================================"

# srun python scripts/main.py \
#             --n_clients 150 \
#             --n_clusters 3 \
#             --n_features $PARAM \
#             --model decision_tree \
#             --lmbd 0.05 \
#             --dataset synthetic \
#             --algo Algorithm2_SKLearn \
#             --R 200 \
#             --S 30 \
#             --fname ${OUT_DIR}/linear_syn_dm/Algorithm2_SKLearn/dt_synthetic_${PARAM}_${SEED}.csv \
#             --device cpu \
#             --problem regression \
#             --seed $SEED

#PARAM_LIST=(0.05 0.1 0.2 0.5) # lmbd
## ===============================
## Index mapping
## ===============================
#IDX=$SLURM_ARRAY_TASK_ID
#
#NP=${#PARAM_LIST[@]} # number of elements in the array
#NS=${#SEEDS[@]}
#
#PARAM_IDX=$(( IDX / NS ))
#SEED_IDX=$(( IDX %  NS ))
#
#PARAM=${PARAM_LIST[$PARAM_IDX]}
#SEED=${SEEDS[$SEED_IDX]}

#srun python scripts/main.py \
#            --n_clients 150 \
#            --n_clusters 3 \
#            --n_features 10 \
#            --model decision_tree \
#            --lmbd $PARAM \
#            --dataset synthetic \
#            --algo Algorithm2_SKLearn \
#            --R 200 \
#            --S 30 \
#            --fname ${OUT_DIR}/linear_syn_lmbd/Algorithm2_SKLearn/dt_synthetic_${PARAM}_${SEED}.csv \
#            --device cpu \
#            --problem regression \
#            --seed $SEED

PARAM_LIST=(10 20 30 50) # S
# ===============================
# Index mapping
# ===============================
IDX=$SLURM_ARRAY_TASK_ID

NP=${#PARAM_LIST[@]} # number of elements in the array
NS=${#SEEDS[@]}

PARAM_IDX=$(( IDX / NS ))
SEED_IDX=$(( IDX %  NS ))

PARAM=${PARAM_LIST[$PARAM_IDX]}
SEED=${SEEDS[$SEED_IDX]}

srun python scripts/main.py \
            --n_clients 150 \
            --n_clusters 3 \
            --n_features 10 \
            --model decision_tree \
            --lmbd 0.05 \
            --dataset synthetic \
            --algo Algorithm2_SKLearn \
            --R 200 \
            --S $PARAM \
            --fname ${OUT_DIR}/linear_syn_S/Algorithm2_SKLearn/dt_synthetic_${PARAM}_${SEED}.csv \
            --device cpu \
            --problem regression \
            --seed $SEED

# S_LIST=(10 20 30 50)
# for S in "${S_LIST[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         echo "========================================"
#         echo "Running experiment: S=$S, seed=$SEED"
#         echo "========================================"

#         START_EXP=$SECONDS

#         python scripts/main.py \
#             --n_clients 100 \
#             --n_clusters 3 \
#             --n_features  \
#             --model decision_tree \
#             --lmbd 0.05 \
#             --dataset synthetic \
#             --algo Algorithm2_SKLearn \
#             --R 200 \
#             --S $S \
#             --fname ${OUT_DIR}/linear_syn_S/Algorithm2_SKLearn/dt_synthetic_${S}_${SEED}.csv \
#             --device cpu \
#             --problem regression \
#             --seed $SEED

#         END_EXP=$SECONDS
#         ELAPSED_EXP=$(( END_EXP - START_EXP ))
#         echo "Experiment completed in $ELAPSED_EXP seconds"
#     done
# done
# END=$SECONDS
# ELAPSED=$(( END - START ))
# echo "All experiments completed in $ELAPSED seconds"


# S_LIST=(10 20 30 50)
# for S in "${S_LIST[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         echo "========================================"
#         echo "Running experiment: S=$S, seed=$SEED"
#         echo "========================================"

#         START_EXP=$SECONDS

#         python scripts/main.py \
#             --n_clients 100 \
#             --n_clusters 3 \
#             --n_features  \
#             --model decision_tree \
#             --lmbd 0.05 \
#             --dataset synthetic \
#             --algo Algorithm2_SKLearn \
#             --R 200 \
#             --S $S \
#             --fname ${OUT_DIR}/linear_syn_S/Algorithm2_SKLearn/dt_synthetic_${S}_${SEED}.csv \
#             --device cpu \
#             --problem regression \
#             --seed $SEED

#         END_EXP=$SECONDS
#         ELAPSED_EXP=$(( END_EXP - START_EXP ))
#         echo "Experiment completed in $ELAPSED_EXP seconds"
#     done
# done
# END=$SECONDS
# ELAPSED=$(( END - START ))
# echo "All experiments completed in $ELAPSED seconds"
