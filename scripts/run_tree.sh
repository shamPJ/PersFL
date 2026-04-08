#!/bin/bash
#SBATCH --job-name=persfl_tree
#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3
#SBATCH --array=0-39
#SBATCH --output=logs/out_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err

set -euo pipefail

# ===============================
# Environment
# ===============================
module load mamba
source activate pytorch-env

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"
export OMP_NUM_THREADS=1

SEEDS=(0 1 2 3 4 5 6 7 8 9)

OUT_DIR="results"
ALG="Algorithm2_SKLearn"

mkdir -p "$OUT_DIR"

EXPS=(
  "linear_syn_dm"
  "linear_syn_S"
  "linear_syn_lmbd"
  "linear_syn_noise"
  "linear_syn_noise_w"
  "linear_syn_nclusters"
)

for EXP in "${EXPS[@]}"; do
    mkdir -p "${OUT_DIR}/${EXP}/${ALG}"
done

# ===============================
# Choose experiment (must match EXPS)
# ===============================
exp="linear_syn_noise_w"

# ===============================
# Parameter selection
# ===============================
case "$exp" in
    linear_syn_dm)
        PARAM_LIST=(2 10 50 100)
        EXTRA_ARGS="--n_features PARAM"
        SUBDIR="linear_syn_dm"
        ;;

    linear_syn_lmbd)
        PARAM_LIST=(0.05 0.1 0.2 0.5)
        EXTRA_ARGS="--lmbd PARAM"
        SUBDIR="linear_syn_lmbd"
        ;;

    linear_syn_S)
        PARAM_LIST=(3 5 10 30)
        EXTRA_ARGS="--S PARAM"
        SUBDIR="linear_syn_S"
        ;;

    linear_syn_noise)
        PARAM_LIST=(0.1 0.5 1 1.5)
        EXTRA_ARGS="--noise_scale PARAM"
        SUBDIR="linear_syn_noise"
        ;;

    linear_syn_noise_w)
        PARAM_LIST=(0.1 0.5 1 1.5)
        EXTRA_ARGS="--noise_weight PARAM --noise_scale 0"
        SUBDIR="linear_syn_noise_w"
        ;;

    linear_syn_nclusters)
        PARAM_LIST=(2 3 5 10)
        EXTRA_ARGS="--n_clusters PARAM --noise_weight 0 --noise_scale 0"
        SUBDIR="linear_syn_nclusters"
        ;;

    *)
        echo "Unknown experiment: $exp"
        exit 1
        ;;
esac

# ===============================
# Index mapping (shared)
# ===============================
IDX=$SLURM_ARRAY_TASK_ID
NP=${#PARAM_LIST[@]}
NS=${#SEEDS[@]}

if (( IDX >= NP * NS )); then
    echo "Index $IDX out of range (max = $((NP*NS-1)))"
    exit 1
fi

PARAM_IDX=$(( IDX / NS ))
SEED_IDX=$(( IDX % NS ))

PARAM=${PARAM_LIST[$PARAM_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# Replace placeholder PARAM in EXTRA_ARGS
EXTRA_ARGS=${EXTRA_ARGS//PARAM/$PARAM}

# ===============================
# Run
# ===============================
echo "========================================"
echo "Experiment: $exp"
echo "Param     : $PARAM"
echo "Seed      : $SEED"
echo "Task ID   : $IDX"
echo "========================================"
# Args from  $EXTRA_ARGS will override earlier ones
srun python scripts/main.py \
    --n_clients 150 \
    --n_clusters 3 \
    --n_features 10 \
    --model decision_tree \
    --lmbd 0.05 \
    --dataset synthetic \
    --algo "$ALG" \
    --R 200 \
    --S 30 \
    $EXTRA_ARGS \
    --fname "${OUT_DIR}/${SUBDIR}/${ALG}/dt_synthetic_${PARAM}_${SEED}.csv" \
    --device cpu \
    --problem regression \
    --seed "$SEED"
