#!/usr/bin/env bash
#SBATCH --job-name=saga
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=saga-%j.out
#SBATCH --error=saga-%j.err
set -euo pipefail
seed=${1:?Usage: sbatch scripts/submit_saga.sh SEED MODEL [RECIPE] [OUTPUT_ROOT]}
model=${2:?Supply an explicit MACE model path}
recipe=${3:-examples/nh3_gpu.json}
root=${4:-runs/nh3_gpu}
[[ -f saga/campaign.py && -f "$recipe" ]] || { echo 'Submit from the repository root with an existing recipe.' >&2; exit 2; }
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1
args=(--seed "$seed" --model "$model" --recipe "$recipe" --root "$root")
[[ -z ${MAX_BATCH_ATOMS:-} ]] || args+=(--max-batch-atoms "$MAX_BATCH_ATOMS")
[[ -z ${MAX_BATCH_EDGES:-} ]] || args+=(--max-batch-edges "$MAX_BATCH_EDGES")
[[ -z ${GENERATIONS:-} ]] || args+=(--generations "$GENERATIONS")
[[ ${DRY_RUN:-0} != 1 ]] || args+=(--dry-run)
"${PYTHON:-python3}" -m saga.campaign "${args[@]}" &
worker=$!
trap 'kill -TERM "$worker" 2>/dev/null || true; wait "$worker" || true; exit 143' TERM INT
wait "$worker"
