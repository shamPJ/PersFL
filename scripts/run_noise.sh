#!/bin/bash
#SBATCH --job-name=persfl_noise_y
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

export PYTHONPATH=$PYTHONPATH:$PWD

# ===============================
# Sweep grids
# ===============================
# Sweep parameters
NOISE_LIST=(0.1 0.5 1 1.5) # label noise dimensionality grid
SEEDS=(0 1 2 3 4 5 6 7 8 9)   # repetitions

# Output directory
OUT_DIR="../results/linear_syn_noise"
mkdir -p $OUT_DIR logs

# Algorithm subdirectories
ALGOS=("Algorithm1" "Algorithm2")
for ALG in "${ALGOS[@]}"; do
    mkdir -p "${OUT_DIR}/${ALG}"
done

# ===============================
# Index mapping
# ===============================
IDX=$SLURM_ARRAY_TASK_ID

NN=${#NOISE_LIST[@]} # number of elements in the array
NS=${#SEEDS[@]}

# Total jobs = 4 * 10 = 40

N_IDX=$(( IDX / NS ))
SEED_IDX=$(( IDX %  NS ))

NOISE=${NOISE_LIST[$N_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "========================================"
echo "Running experiment on GPU"
echo "  noise   = $NOISE"
echo "  seed    = $SEED"
echo "  GPU     = $CUDA_VISIBLE_DEVICES"
echo "========================================"

# ===============================
# Run experiment
# ===============================
# srun python scripts/main.py \
#     --n_clients 150 \
#     --n_clusters 3 \
#     --n_features 10 \
#     --model linreg \
#     --dataset synthetic \
#     --noise_scale $NOISE \
#     --algo Algorithm1 \
#     --R 1500 \
#     --R_local 0 \
#     --lrate 0.01 \
#     --S 30 \
#     --fname ${OUT_DIR}/Algorithm1/linear_syn_noise_${NOISE}_${SEED}.csv \
#     --device cuda \
#     --problem regression \
#     --seed $SEED 

srun python scripts/main.py \
    --n_clients 150 \
    --n_clusters 3 \
    --n_features 10 \
    --model linreg \
    --dataset synthetic \
    --noise_scale $NOISE \
    --algo Algorithm2 \
    --R 1500 \
    --R_local 0 \
    --lrate 0.01 \
    --S 30 \
    --fname ${OUT_DIR}/Algorithm2/linear_syn_noise_${NOISE}_${SEED}.csv \
    --device cuda \
    --problem regression \
    --seed $SEED 





