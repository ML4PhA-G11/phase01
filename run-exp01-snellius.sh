#!/usr/bin/env bash
# =============================================================================
# Exp01 end-to-end runner — Karman vortex street, BGK, GAVG (D4-equivariant)
# =============================================================================
# Reproduces every step of task01-01-ready-to-execute.md on Snellius:
#   data   (step 1+2) — check, and if missing (re)generate, the simulator data
#   train  (step 3)   — GPU pre-flight + train the D4-equivariant model
#   apply  (step 4)   — apply the trained model back to the Karman simulator
#
# Usage:
#   ./run-exp01-snellius.sh                  # default: all stages
#   ./run-exp01-snellius.sh all              # same as above
#   ./run-exp01-snellius.sh data             # only step 1+2
#   ./run-exp01-snellius.sh train            # only step 3 (GPU check + train)
#   ./run-exp01-snellius.sh apply            # only step 4
#   ./run-exp01-snellius.sh data train       # any subset, in order data→train→apply
#
# This drives the commands directly. To run under SLURM, wrap the invocation,
# e.g.:  sbatch --time=08:00:00 --gpus=1 run-exp01-snellius.sh train
#
# Override any path/parameter via the environment, e.g.:
#   DATA_DIR=/path/to/data SAMPLES_PER_STEP=4000 ./run-exp01-snellius.sh
#   MODEL_PATH=/path/to/model.keras ./run-exp01-snellius.sh apply
# =============================================================================
set -euo pipefail

# --- Stage selection (positional args) --------------------------------------
RUN_DATA=false
RUN_TRAIN=false
RUN_APPLY=false

if (( $# == 0 )); then
    RUN_DATA=true; RUN_TRAIN=true; RUN_APPLY=true
else
    for stage in "$@"; do
        case "$stage" in
            all)              RUN_DATA=true;  RUN_TRAIN=true; RUN_APPLY=true ;;
            data|step1|step2) RUN_DATA=true ;;
            train|step3)      RUN_TRAIN=true ;;
            apply|step4)      RUN_APPLY=true ;;
            -h|--help)
                sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
                exit 0
                ;;
            *)
                echo "ERROR: unknown stage '$stage'. Valid: data | train | apply | all" >&2
                exit 2
                ;;
        esac
    done
fi

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
if $RUN_DATA; then
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
else
    log "Skipping data stage (step 1+2)."
fi

# =============================================================================
# Step 3. Train the D4-equivariant model on the simulator data
# =============================================================================
if $RUN_TRAIN; then
    # First: verify TF can actually see and use the GPU in this environment, so
    # we fail fast here instead of hours into training on CPU.
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
else
    log "Skipping train stage (step 3)."
fi

# =============================================================================
# Step 4. Apply the trained model back to the Karman simulator
# =============================================================================
if $RUN_APPLY; then
    # Resolve the model to apply. Honour an explicit MODEL_PATH override (handy
    # when re-running just the apply stage); otherwise pick the newest
    # model.keras under the karman run dirs.
    if [[ -z "${MODEL_PATH:-}" ]]; then
        log "Locating the trained model"
        MODEL_PATH="$(
            find "${ME_DIR}/artifacts-run-all-tensorflow" \
                -type f -name model.keras -path '*karman*' \
                -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr | head -1 | cut -d' ' -f2-
        )"
    fi
    if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}" ]]; then
        echo "ERROR: no trained model.keras found under ${ME_DIR}/artifacts-run-all-tensorflow" >&2
        echo "       (set MODEL_PATH=/path/to/model.keras to override.)" >&2
        exit 1
    fi
    log "Trained model: ${MODEL_PATH}"

    log "Step 4: applying model (branch ${BRANCH})"
    (
        cd "${APPLY_DIR}"
        git checkout "${BRANCH}"
        uv run python apply-nn.py \
            --animate \
            --model-path "${MODEL_PATH}" \
            --data-dir "${DATA_DIR}" \
            --out-dir "${EVAL_OUT_DIR}"
    )

    log "Exp01 complete. Evaluation results -> ${EVAL_OUT_DIR}"
else
    log "Skipping apply stage (step 4)."
fi
