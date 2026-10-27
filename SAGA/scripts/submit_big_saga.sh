#!/usr/bin/env bash
#SBATCH --job-name=saga-nh3-big
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=saga-big-%j.out
#SBATCH --error=saga-big-%j.err
set -euo pipefail
seed=${1:?Usage: sbatch scripts/submit_big_saga.sh SEED [MODEL] [OUTPUT_ROOT]}
model=${2:-${MODEL:-models/s8.model}}
root=${3:-${RUN_ROOT:-big_nh3_runs}}
[[ -f saga/campaign.py && -f campaign.json ]] || { echo 'Submit from the extracted campaign directory.' >&2; exit 2; }
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1
args=(--seed "$seed" --model "$model" --root "$root" --recipe campaign.json
      --max-batch-atoms "${MAX_BATCH_ATOMS:-512}" --max-batch-edges "${MAX_BATCH_EDGES:-100000}")
[[ -z ${GENERATIONS:-} ]] || args+=(--generations "$GENERATIONS")
[[ ${DRY_RUN:-0} != 1 ]] || args+=(--dry-run)
"${PYTHON:-python3}" -m saga.campaign "${args[@]}" &
worker=$!
trap 'kill -TERM "$worker" 2>/dev/null || true; wait "$worker" || true; exit 143' TERM INT
wait "$worker"
