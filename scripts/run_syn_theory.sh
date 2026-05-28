#!/bin/bash
#SBATCH --job-name=persfl_syn_theory
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3
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
M_LIST=(5 10 50 100)
NOISE_LIST=(0.1 1.0 1.5 5)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# Output directory
OUT_DIR="results/linear_syn_theory/Algorithm1"
mkdir -p $OUT_DIR

# ===============================
# Index mapping
# Total jobs = 4 * 10 = 40  (noise looped inside each job)
# ===============================
IDX=$SLURM_ARRAY_TASK_ID

NM=${#M_LIST[@]}
NS=${#SEEDS[@]}

SEED_IDX=$(( IDX % NS ))
M_IDX=$(( IDX / NS ))

M=${M_LIST[$M_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "========================================"
echo "Running experiment on GPU"
echo "  M    = $M"
echo "  seed = $SEED"
echo "  GPU  = $CUDA_VISIBLE_DEVICES"
echo "========================================"

# ===============================
# Loop over noise levels
# ===============================
for NOISE in "${NOISE_LIST[@]}"; do
    echo "  noise_scale = $NOISE"
    srun python scripts/main.py \
        --n_clients 150 \
        --n_clusters 3 \
        --n_samples $M \
        --n_features 10 \
        --no_scale \
        --model linreg \
        --dataset synthetic \
        --algo Algorithm1 \
        --R 300 \
        --R_local 1 \
        --lrate 0.01 \
        --S 5 \
        --noise_scale $NOISE \
        --fname ${OUT_DIR}/linear_syn_m${M}_noise${NOISE}_seed${SEED}.csv \
        --device cuda \
        --problem regression \
        --seed $SEED
done
