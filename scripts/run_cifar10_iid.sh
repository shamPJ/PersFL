#!/bin/bash
#SBATCH --job-name=persfl_dm
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=3 # task is job instance created from the array; each task runs .sh independently
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --output=logs/out_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err

# ===============================
# Environment
# ===============================
module load mamba
source activate pytorch-env-cuda118

# ===============================
# Sweep grids
# ===============================
SEEDS=(0 1 2 3 4 5 6 7 8 9)  # repetitions

export PYTHONPATH=$PYTHONPATH:$PWD

# Output directory
OUT_DIR="results/cnn_cifar10_iid"
mkdir -p $OUT_DIR

# Algorithm subdirectories
ALGOS=("Algorithm1" "Algorithm2" "FedAvg")
for ALG in "${ALGOS[@]}"; do
    mkdir -p "${OUT_DIR}/${ALG}"
done

# ===============================
# Index mapping
# ===============================
SEED=$SLURM_ARRAY_TASK_ID

echo "========================================"
echo "Running experiment on GPU"
echo "  seed    = $SEED"
echo "  GPU     = $CUDA_VISIBLE_DEVICES"
echo "========================================"

# ===============================
# Run experiment
# ===============================
srun python scripts/main.py \
    --n_clients 10 \
    --n_clusters 2 \
    --n_classes 10 \
    --n_samples 500 \
    --n_samples_val 1000 \
    --model cnn \
    --dataset cifar10 \
    --algo Algorithm1 \
    --R 50 \
    --R_local 5 \
    --lrate 0.01 \
    --lrate_decay 0.9 \
    --S 5 \
    --fname ${OUT_DIR}/Algorithm1/cnn_cifar10_iid_${SEED}.csv \
    --device cuda \
    --problem classification \
    --seed $SEED 
    
# srun python main.py \
#     --n_clients 150 \
#     --n_clusters 3 \
#     --n_features $D \
#     --model linreg \
#     --dataset synthetic \
#     --algo Algorithm2 \
#     --R 1500 \
#     --R_local 0 \
#     --lrate 0.01 \
#     --S 30 \
#     --fname ${OUT_DIR}/Algorithm2/linear_syn_dm_${D}_${SEED}.csv \
#     --device cuda \
#     --problem regression \
#     --seed $SEED 

srun python scripts/main.py \
        --n_clients 10 \
        --n_clusters 2 \
        --n_classes 10 \
	--n_samples 500 \
    	--n_samples_val 1000 \
	--model cnn \
        --dataset cifar10 \
        --algo FedAvg \
        --R 50 \
        --R_local 5 \
        --lrate 0.01 \
        --lrate_decay 0.9 \
        --S 9 \
        --fname ${OUT_DIR}/FedAvg/cnn_cifar10_iid_${SEED}.csv \
        --device cuda \
        --problem classification \
        --seed $SEED 





