#!/usr/bin/env bash
# =============================================================================
# Exp01 end-to-end runner — Karman vortex street, BGK, GAVG (D4-equivariant)
# =============================================================================
# Reproduces every step of task01-01-ready-to-execute.md on Snellius:
#   1. Check whether the simulator data already exists on /gpfs scratch.
#   2. If not, (re)generate it with the simulator.
#   3. Train the D4-equivariant collision operator on that data.
#   4. Apply the trained model back to the Karman simulator.
#
# This drives the commands directly. To run under SLURM, wrap the invocation,
# e.g.:  sbatch --time=08:00:00 --gpus=1 run-exp01-snellius.sh
#
# Override any path/parameter via the environment, e.g.:
#   DATA_DIR=/path/to/data SAMPLES_PER_STEP=4000 ./run-exp01-snellius.sh
# =============================================================================
set -euo pipefail

# --- Resolve repository paths relative to this script (the phase01 glue dir) --
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR="${SIM_DIR:-${HERE}/../simulators}"
ME_DIR="${ME_DIR:-${HERE}/../model-experiments}"
APPLY_DIR="${APPLY_DIR:-${HERE}/../Apply-NN-KarmanVortexStreet}"

# --- Experiment configuration ------------------------------------------------
BRANCH="${BRANCH:-dev-C04-helper-scripts}"
DATA_DIR="${DATA_DIR:-/gpfs/scratch1/shared/scur0076/output-lbm-data-02-30000steps-data.per.step-npy}"
SAVE_EVERY="${SAVE_EVERY:-1}"          # simulator dump cadence (step 2)
N_STEPS="${N_STEPS:-30000}"            # simulator timesteps (step 2)

# Training sub-sampling (passed through to train-d4equivariant-karman.sh / run_all.py)
export DATA_DIR
export SAMPLES_PER_STEP="${SAMPLES_PER_STEP:-2000}"
export STEP_STRIDE="${STEP_STRIDE:-10}"
export MAX_STEPS="${MAX_STEPS:-}"

# Where the apply step writes its evaluation results.
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${APPLY_DIR}/eval_results}"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# --- Optional Snellius environment setup ------------------------------------
# Load modules / activate an environment here if your site needs it, e.g.:
#   module load 2023; module load Python/3.11.3-GCCcore-12.3.0
# Left as a hook; uncomment/edit as appropriate for your account.
module load 2025
module load CUDA/12.8.0
if [[ -n "${SNELLIUS_MODULES:-}" ]]; then
    log "Loading modules: ${SNELLIUS_MODULES}"
    # shellcheck disable=SC1090
    module load ${SNELLIUS_MODULES}
fi

# =============================================================================
# Step 1 + 2. Ensure the simulator data exists
# =============================================================================
log "Step 1: checking for simulator data in ${DATA_DIR}"

shopt -s nullglob
existing=( "${DATA_DIR}"/fpre_*.npy )
shopt -u nullglob

if (( ${#existing[@]} > 0 )); then
    log "Found ${#existing[@]} fpre_*.npy files — using existing data (skipping generation)."
else
    log "Step 2: no data found — regenerating with the simulator (${N_STEPS} steps, --save-every ${SAVE_EVERY})."
    mkdir -p "${DATA_DIR}"
    (
        cd "${SIM_DIR}"
        python lbm_karman-ng.py \
            --n-steps "${N_STEPS}" \
            --save-every "${SAVE_EVERY}" \
            --out-dir "${DATA_DIR}"
    )
    log "Data generation complete -> ${DATA_DIR}"
fi

# =============================================================================
# Step 3. Train the D4-equivariant model on the simulator data
# =============================================================================
# First: verify TF can actually see and use the GPU in this environment, so we
# fail fast here instead of hours into training on CPU.
log "Step 3a: verifying TensorFlow GPU availability"
(
    cd "${ME_DIR}"
    git checkout "${BRANCH}"
    bash scripts/cuda-gpu-tensorflow-enabled.sh
)

log "Step 3b: training (branch ${BRANCH}) with DATA_DIR=${DATA_DIR}"
(
    cd "${ME_DIR}"
    scripts/train-d4equivariant-karman.sh
)

# Locate the model just produced: newest model.keras under the karman run dirs.
log "Locating the trained model"
MODEL_PATH="$(
    find "${ME_DIR}/artifacts-run-all-tensorflow" \
        -type f -name model.keras -path '*karman*' \
        -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d' ' -f2-
)"
if [[ -z "${MODEL_PATH}" ]]; then
    echo "ERROR: no trained model.keras found under ${ME_DIR}/artifacts-run-all-tensorflow" >&2
    exit 1
fi
log "Trained model: ${MODEL_PATH}"

# =============================================================================
# Step 4. Apply the trained model back to the Karman simulator
# =============================================================================
log "Step 4: applying model (branch ${BRANCH})"
(
    cd "${APPLY_DIR}"
    git checkout "${BRANCH}"
    uv run python apply-nn.py \
        --model-path "${MODEL_PATH}" \
        --data-dir "${DATA_DIR}" \
        --out-dir "${EVAL_OUT_DIR}"
)

log "Exp01 complete. Evaluation results -> ${EVAL_OUT_DIR}"
