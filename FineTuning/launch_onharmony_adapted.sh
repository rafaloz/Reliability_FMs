#!/usr/bin/env bash
# Run all seven retained adapted-model families on ON-Harmony in series.

set -euo pipefail

GPU="${1:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MYLOCAL_PYTHON="${MYLOCAL_PYTHON:-/home/export/personal/rnavgon/.conda/envs/myLocalEnv/bin/python}"
NEUROVFM_PYTHON="${NEUROVFM_PYTHON:-/home/export/personal/rnavgon/.conda/envs/neurovfm/bin/python}"
ADAPTED_ROOT="${PROJECT_ROOT}/Results/brain_age/adapted"
OUTPUT_ROOT="${PROJECT_ROOT}/Results/brain_age/onharmony/adapted"
LOG_ROOT="${PROJECT_ROOT}/Results/brain_age/logs/onharmony_adapted"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

for executable in "${MYLOCAL_PYTHON}" "${NEUROVFM_PYTHON}"; do
    if [[ ! -x "${executable}" ]]; then
        echo "Required Python executable is unavailable: ${executable}" >&2
        exit 1
    fi
done

if command -v klist >/dev/null 2>&1; then
    if [[ -z "${KRB5CCNAME:-}" ]] || ! klist -s; then
        echo "No valid inherited Kerberos cache; refusing a long NFS run." >&2
        exit 1
    fi
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
mkdir -p "/tmp/mpl-onharmony-${USER}" "/tmp/triton-neurovfm-${USER}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR="/tmp/mpl-onharmony-${USER}"
export TRITON_CACHE_DIR="/tmp/triton-neurovfm-${USER}"
export PYTHONUNBUFFERED=1

echo "Adapted ON-Harmony inference: seven retained model families"
echo "  project       : ${PROJECT_ROOT}"
echo "  physical GPU  : ${GPU}"
echo "  Kerberos cache: ${KRB5CCNAME:-not exported}"
echo "  outputs       : ${OUTPUT_ROOT}"

audit_model() {
    local python_executable="$1"
    local model_name="$2"
    local script="$3"
    local run_name="$4"
    local selection_name="$5"
    local log_path="${LOG_ROOT}/${model_name}_${run_name}_audit_${RUN_ID}.log"

    echo "[$(date --iso-8601=seconds)] AUDIT ${model_name}/${run_name}"
    "${python_executable}" -u "${script}" audit \
        --run-dir "${ADAPTED_ROOT}/${model_name}/${run_name}" \
        --selection-dir "${ADAPTED_ROOT}/${model_name}/${selection_name}" \
        --subject sub-03286 \
        --session NOT1ACH001 \
        --check-input-files \
        --inspect-model \
        --smoke-forward \
        --device cuda \
        --amp bfloat16 \
        2>&1 | tee "${log_path}"
}

predict_model() {
    local python_executable="$1"
    local model_name="$2"
    local script="$3"
    local run_name="$4"
    local selection_name="$5"
    local output_dir="${OUTPUT_ROOT}/${model_name}/${run_name}"
    local log_path="${LOG_ROOT}/${model_name}_${run_name}_${RUN_ID}.log"

    if [[ -e "${output_dir}" ]]; then
        echo "Output already exists; refusing to reuse or overwrite it: ${output_dir}" >&2
        exit 1
    fi

    echo "[$(date --iso-8601=seconds)] START ${model_name}/${run_name}"
    "${python_executable}" -u "${script}" predict \
        --run-dir "${ADAPTED_ROOT}/${model_name}/${run_name}" \
        --selection-dir "${ADAPTED_ROOT}/${model_name}/${selection_name}" \
        --device cuda \
        --amp bfloat16 \
        --output-dir "${output_dir}" \
        2>&1 | tee "${log_path}"
    echo "[$(date --iso-8601=seconds)] DONE  ${model_name}/${run_name}"
}

declare -A PREFLIGHT_OUTPUTS=()
PREFLIGHT_COUNT=0

require_file() {
    local label="$1"
    local path="$2"
    if [[ ! -f "${path}" ]]; then
        echo "Missing ${label}: ${path}" >&2
        exit 1
    fi
}

preflight_prediction() {
    local python_executable="$1"
    local model_name="$2"
    local script="$3"
    local run_name="$4"
    local selection_name="$5"
    local run_dir="${ADAPTED_ROOT}/${model_name}/${run_name}"
    local selection_dir="${ADAPTED_ROOT}/${model_name}/${selection_name}"
    local output_dir="${OUTPUT_ROOT}/${model_name}/${run_name}"

    if [[ ! -x "${python_executable}" ]]; then
        echo "Python executable is unavailable: ${python_executable}" >&2
        exit 1
    fi
    require_file "prediction entry point" "${PROJECT_ROOT}/${script}"
    require_file "adapted checkpoint" "${run_dir}/selected_model.pt"
    require_file "immutable training lock" "${run_dir}/LOCKED.json"
    require_file "validation-only LR selection" "${selection_dir}/SELECTED_LR.json"

    if [[ -n "${PREFLIGHT_OUTPUTS[${output_dir}]:-}" ]]; then
        echo "Duplicate prediction target in launcher: ${output_dir}" >&2
        exit 1
    fi
    PREFLIGHT_OUTPUTS["${output_dir}"]=1
    if [[ -e "${output_dir}" ]]; then
        echo "Output already exists; refusing to start: ${output_dir}" >&2
        exit 1
    fi
    PREFLIGHT_COUNT=$((PREFLIGHT_COUNT + 1))
}

preflight_three_seeds() {
    local python_executable="$1"
    local model_name="$2"
    local script="$3"
    local selected_lr="$4"
    local selection_name="$5"

    preflight_prediction "${python_executable}" "${model_name}" "${script}" \
        "selection_seed0_${selected_lr}" "${selection_name}"
    preflight_prediction "${python_executable}" "${model_name}" "${script}" \
        "final_seed1_${selected_lr}" "${selection_name}"
    preflight_prediction "${python_executable}" "${model_name}" "${script}" \
        "final_seed2_${selected_lr}" "${selection_name}"
}

run_three_seeds() {
    local python_executable="$1"
    local model_name="$2"
    local script="$3"
    local selected_lr="$4"
    local selection_name="$5"

    audit_model "${python_executable}" "${model_name}" "${script}" \
        "selection_seed0_${selected_lr}" "${selection_name}"
    predict_model "${python_executable}" "${model_name}" "${script}" \
        "selection_seed0_${selected_lr}" "${selection_name}"
    predict_model "${python_executable}" "${model_name}" "${script}" \
        "final_seed1_${selected_lr}" "${selection_name}"
    predict_model "${python_executable}" "${model_name}" "${script}" \
        "final_seed2_${selected_lr}" "${selection_name}"
}

echo "[$(date --iso-8601=seconds)] PREFLIGHT: validating all retained artifacts and outputs"

preflight_three_seeds "${MYLOCAL_PYTHON}" "3DNeuroSimCLR" \
    "FineTuning/3DNeuroSimCLR/predict_onharmony.py" "intermediate" "lr_selection_seed0"
preflight_three_seeds "${MYLOCAL_PYTHON}" "Curia2" \
    "FineTuning/Curia2/predict_onharmony.py" "high" "lr_selection_seed0"
preflight_three_seeds "${MYLOCAL_PYTHON}" "BrainIAC" \
    "FineTuning/BrainIAC/predict_onharmony.py" "high" "lr_selection_seed0"
preflight_three_seeds "${MYLOCAL_PYTHON}" "BrainSegFounder" \
    "FineTuning/BrainSegFounder/predict_onharmony.py" "high" "lr_selection_seed0"

for run_name in \
    fold0_seed0_high fold0_seed1_high fold0_seed2_high \
    fold1_seed0_high fold2_seed0_high fold3_seed0_high fold4_seed0_high; do
    preflight_prediction "${MYLOCAL_PYTHON}" "AnatCL" \
        "FineTuning/AnatCL/predict_onharmony.py" \
        "${run_name}" "fold0_lr_selection"
done

preflight_three_seeds "${MYLOCAL_PYTHON}" "yAware" \
    "FineTuning/yAware/predict_onharmony.py" "high" "lr_selection_seed0"
preflight_three_seeds "${NEUROVFM_PYTHON}" "NeuroVFM" \
    "FineTuning/NeuroVFM/predict_onharmony.py" "high" "lr_selection_seed0"

if [[ "${PREFLIGHT_COUNT}" -ne 25 ]]; then
    echo "Internal launcher error: expected 25 targets, found ${PREFLIGHT_COUNT}" >&2
    exit 1
fi
echo "[$(date --iso-8601=seconds)] PREFLIGHT PASS: ${PREFLIGHT_COUNT} fresh targets"

run_three_seeds "${MYLOCAL_PYTHON}" "3DNeuroSimCLR" \
    "FineTuning/3DNeuroSimCLR/predict_onharmony.py" "intermediate" "lr_selection_seed0"

run_three_seeds "${MYLOCAL_PYTHON}" "Curia2" \
    "FineTuning/Curia2/predict_onharmony.py" "high" "lr_selection_seed0"

run_three_seeds "${MYLOCAL_PYTHON}" "BrainIAC" \
    "FineTuning/BrainIAC/predict_onharmony.py" "high" "lr_selection_seed0"

run_three_seeds "${MYLOCAL_PYTHON}" "BrainSegFounder" \
    "FineTuning/BrainSegFounder/predict_onharmony.py" "high" "lr_selection_seed0"

audit_model "${MYLOCAL_PYTHON}" "AnatCL" \
    "FineTuning/AnatCL/predict_onharmony.py" \
    "fold0_seed0_high" "fold0_lr_selection"
for run_name in \
    fold0_seed0_high fold0_seed1_high fold0_seed2_high \
    fold1_seed0_high fold2_seed0_high fold3_seed0_high fold4_seed0_high; do
    predict_model "${MYLOCAL_PYTHON}" "AnatCL" \
        "FineTuning/AnatCL/predict_onharmony.py" \
        "${run_name}" "fold0_lr_selection"
done

run_three_seeds "${MYLOCAL_PYTHON}" "yAware" \
    "FineTuning/yAware/predict_onharmony.py" "high" "lr_selection_seed0"

echo "[$(date --iso-8601=seconds)] CACHE NeuroVFM ON-Harmony inputs"
"${NEUROVFM_PYTHON}" -u FineTuning/NeuroVFM/predict_onharmony.py cache \
    2>&1 | tee "${LOG_ROOT}/NeuroVFM_cache_${RUN_ID}.log"

run_three_seeds "${NEUROVFM_PYTHON}" "NeuroVFM" \
    "FineTuning/NeuroVFM/predict_onharmony.py" "high" "lr_selection_seed0"

echo "[$(date --iso-8601=seconds)] COMPLETE: seven families, 25 adapted checkpoints"
