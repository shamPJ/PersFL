#!/bin/bash
#SBATCH --job-name=hetfl_gmm
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2 # task is job instance created from the array; each task runs .sh independently
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
D_LIST=(2 10 50 100)         # data dimensionality grid
SEEDS=(0 1 2 3 4 5 6 7 8 9)  # repetitions

# Output directory
OUT_DIR="results/linear_syn_dm"
mkdir -p $OUT_DIR logs

# ===============================
# Index mapping
# ===============================
IDX=$SLURM_ARRAY_TASK_ID

ND=${#D_LIST[@]} # number of elements in the array
NS=${#SEEDS[@]}

# Total jobs = 4 * 10 = 40

D_IDX=$(( IDX / NS ))
SEED_IDX=$(( IDX %  NS ))

D=${D_LIST[$D_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "========================================"
echo "Running experiment on GPU"
echo "  D       = $D"
echo "  seed    = $SEED"
echo "  GPU     = $CUDA_VISIBLE_DEVICES"
echo "========================================"

# ===============================
# Run experiment
# ===============================
srun python main.py \
    --n_clients 150 \
    --n_clusters 3 \
    --n_features $D \
    --model linreg \
    --dataset synthetic \
    --algo persfl \
    --R 1500 \
    --lrate 0.01 \
    --S 30 \
    --fname ${OUT_DIR}/linear_syn_dm_${D}_${SEED}.csv \
    --device cuda \
    --problem regression \
    --seed $SEED 





