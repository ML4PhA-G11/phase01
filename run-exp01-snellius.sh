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
#   ./run-exp01-snellius.sh --dry-run        # show resolved paths + what would run
#
# This drives the commands directly. To run under SLURM, wrap the invocation,
# e.g.:  sbatch --time=08:00:00 --gpus=1 run-exp01-snellius.sh train
#
# Override any path/parameter via the environment, e.g.:
#   DATA_DIR=/path/to/data SAMPLES_PER_STEP=4000 ./run-exp01-snellius.sh
#   MODEL_PATH=/path/to/model.keras ./run-exp01-snellius.sh apply
# =============================================================================
set -euo pipefail

# --- Argument parsing -------------------------------------------------------
RUN_DATA=false
RUN_TRAIN=false
RUN_APPLY=false
DRY_RUN=false
positional=()

for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) DRY_RUN=true ;;
        -h|--help)
            sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        -*)
            echo "ERROR: unknown option '$arg'" >&2
            exit 2
            ;;
        *) positional+=("$arg") ;;
    esac
done

if (( ${#positional[@]} == 0 )); then
    RUN_DATA=true; RUN_TRAIN=true; RUN_APPLY=true
else
    for stage in "${positional[@]}"; do
        case "$stage" in
            all)              RUN_DATA=true;  RUN_TRAIN=true; RUN_APPLY=true ;;
            data|step1|step2) RUN_DATA=true ;;
            train|step3)      RUN_TRAIN=true ;;
            apply|step4)      RUN_APPLY=true ;;
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

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
dry()  { printf '   \033[33m[dry-run]\033[0m %s\n' "$*"; }

# Run a command, or just echo it in dry-run mode. Use for any side-effecting
# command (mkdir, module load, ...).
maybe() {
    if $DRY_RUN; then dry "$*"; else "$@"; fi
}

# Run a command inside a subshell rooted at <cwd>, or echo the equivalent
# in dry-run mode. Use for heavy stage commands (python, training, apply).
exec_in() {
    local cwd="$1"; shift
    if $DRY_RUN; then
        dry "(cd $cwd && $*)"
    else
        ( cd "$cwd" && "$@" )
    fi
}

# --- Plan summary (printed in both real and dry runs) -----------------------
on_off() { $1 && echo "✓" || echo "·"; }

cat <<EOF

============================================================
 Exp01 plan $($DRY_RUN && echo '(DRY RUN — no side effects)')
============================================================
 Stages:     $(on_off $RUN_DATA) data   $(on_off $RUN_TRAIN) train   $(on_off $RUN_APPLY) apply
 Branch:     ${BRANCH}
 Paths:
   phase01 (HERE)    : ${HERE}
   simulator         : ${SIM_DIR}
   model-experiments : ${ME_DIR}
   apply             : ${APPLY_DIR}
   DATA_DIR          : ${DATA_DIR}
   EVAL_OUT_DIR      : ${EVAL_OUT_DIR}
   MODEL_PATH        : ${MODEL_PATH:-<auto: newest model.keras under ${ME_DIR}/artifacts-run-all-tensorflow/*karman*>}
 Sim params:  N_STEPS=${N_STEPS}  SAVE_EVERY=${SAVE_EVERY}
 Train subs.: SAMPLES_PER_STEP=${SAMPLES_PER_STEP}  STEP_STRIDE=${STEP_STRIDE}  MAX_STEPS=${MAX_STEPS:-<none>}
============================================================
EOF

# --- Optional Snellius environment setup ------------------------------------
# Load modules / activate an environment here if your site needs it, e.g.:
#   module load 2023; module load Python/3.11.3-GCCcore-12.3.0
# Left as a hook; uncomment/edit as appropriate for your account.
maybe module load 2025
maybe module load CUDA/12.8.0
if [[ -n "${SNELLIUS_MODULES:-}" ]]; then
    log "Loading modules: ${SNELLIUS_MODULES}"
    # shellcheck disable=SC1090
    maybe module load ${SNELLIUS_MODULES}
fi

# =============================================================================
# Step 1 + 2. Ensure the simulator data exists
# =============================================================================
if $RUN_DATA; then
    log "Step 1: checking for simulator data in ${DATA_DIR}"

    # Glob is read-only — always do it, even in dry-run, so the report below
    # accurately reflects whether step 2 would actually generate anything.
    shopt -s nullglob
    existing=( "${DATA_DIR}"/fpre_*.npy )
    shopt -u nullglob

    if (( ${#existing[@]} > 0 )); then
        log "Found ${#existing[@]} fpre_*.npy files — using existing data (skipping generation)."
    else
        log "Step 2: no data found — would regenerate with the simulator (${N_STEPS} steps, --save-every ${SAVE_EVERY})."
        maybe mkdir -p "${DATA_DIR}"
        exec_in "${SIM_DIR}" python lbm_karman-ng.py \
            --n-steps "${N_STEPS}" \
            --save-every "${SAVE_EVERY}" \
            --out-dir "${DATA_DIR}"
        $DRY_RUN || log "Data generation complete -> ${DATA_DIR}"
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
    exec_in "${ME_DIR}" git checkout "${BRANCH}"
    exec_in "${ME_DIR}" bash scripts/cuda-gpu-tensorflow-enabled.sh

    log "Step 3b: training (branch ${BRANCH}) with DATA_DIR=${DATA_DIR}"
    exec_in "${ME_DIR}" scripts/train-d4equivariant-karman.sh
else
    log "Skipping train stage (step 3)."
fi

# =============================================================================
# Step 4. Apply the trained model back to the Karman simulator
# =============================================================================
if $RUN_APPLY; then
    # Resolve the model to apply. Honour an explicit MODEL_PATH override (handy
    # when re-running just the apply stage); otherwise pick the newest
    # model.keras under the karman run dirs. The find is read-only, so we run
    # it in dry-run too — useful to show which artifact would be picked up.
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
        msg="no trained model.keras found under ${ME_DIR}/artifacts-run-all-tensorflow"
        if $DRY_RUN; then
            log "WARN: ${msg} — apply would fail at runtime unless step 3 produces one first."
            MODEL_PATH="<unresolved>"
        else
            echo "ERROR: ${msg}" >&2
            echo "       (set MODEL_PATH=/path/to/model.keras to override.)" >&2
            exit 1
        fi
    fi
    log "Trained model: ${MODEL_PATH}"

    log "Step 4: applying model (branch ${BRANCH})"
    exec_in "${APPLY_DIR}" git checkout "${BRANCH}"
    exec_in "${APPLY_DIR}" uv run python apply-nn.py \
        --animate \
        --model-path "${MODEL_PATH}" \
        --data-dir "${DATA_DIR}" \
        --out-dir "${EVAL_OUT_DIR}"

    $DRY_RUN || log "Exp01 complete. Evaluation results -> ${EVAL_OUT_DIR}"
else
    log "Skipping apply stage (step 4)."
fi

$DRY_RUN && log "Dry run finished — no side effects performed."
