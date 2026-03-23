#!/bin/bash
#SBATCH --job-name=persfl_S
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3 # task is job instance created from the array; each task runs .sh independently
#SBATCH --gres=gpu:1
#SBATCH --array=0-39
#SBATCH --output=logs/out_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err

# ===============================
# Environment
# ===============================
module load mamba
source activate pytorch-env

# ===============================
# Sweep grids
# ===============================
# Sweep parameters
PARAM_LIST=(10 20 30 50)         # candidate set size grid
SEEDS=(0 1 2 3 4 5 6 7 8 9)  # repetitions

# Output directory
OUT_DIR="../results/linear_syn_S"
mkdir -p $OUT_DIR

# Algorithm subdirectories
ALGOS=("Algorithm1" "Algorithm2")
for ALG in "${ALGOS[@]}"; do
    mkdir -p "${OUT_DIR}/${ALG}"
done

# ===============================
# Index mapping
# ===============================
IDX=$SLURM_ARRAY_TASK_ID

NP=${#PARAM_LIST[@]} # number of elements in the array
NS=${#SEEDS[@]}

# Total jobs = 4 * 10 = 40

PARAM_IDX=$(( IDX / NS ))
SEED_IDX=$(( IDX %  NS ))

PARAM=${PARAM_LIST[$PARAM_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "========================================"
echo "Running experiment on GPU"
echo "  S       = $PARAM"
echo "  seed    = $SEED"
echo "  GPU     = $CUDA_VISIBLE_DEVICES"
echo "========================================"

# ===============================
# Run experiment
# ===============================
# srun python main.py \
#     --n_clients 150 \
#     --n_clusters 3 \
#     --n_features 10 \
#     --model linreg \
#     --dataset synthetic \
#     --algo Algorithm1 \
#     --R 1500 \
#     --R_local 0 \
#     --lrate 0.01 \
#     --S $PARAM \
#     --fname ${OUT_DIR}/Algorithm1/linear_syn_S_${PARAM}_${SEED}.csv \
#     --device cuda \
#     --problem regression \
#     --seed $SEED 

srun python main.py \
    --n_clients 150 \
    --n_clusters 3 \
    --n_features 10 \
    --model linreg \
    --dataset synthetic \
    --algo Algorithm2 \
    --R 1500 \
    --R_local 0 \
    --lrate 0.01 \
    --S $PARAM \
    --fname ${OUT_DIR}/Algorithm2/linear_syn_S_${PARAM}_${SEED}.csv \
    --device cuda \
    --problem regression \
    --seed $SEED 





