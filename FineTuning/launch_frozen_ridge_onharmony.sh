#!/usr/bin/env bash
# Fit 11 frozen Ridge heads, freeze ON-Harmony predictions, then evaluate test.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MYLOCAL_PYTHON="${MYLOCAL_PYTHON:-/home/export/personal/rnavgon/.conda/envs/myLocalEnv/bin/python}"
NEUROVFM_PYTHON="${NEUROVFM_PYTHON:-/home/export/personal/rnavgon/.conda/envs/neurovfm/bin/python}"
EMBEDDING_ROOT="${PROJECT_ROOT}/Results/brain_age/embeddings_FT"
RIDGE_ROOT="${PROJECT_ROOT}/Results/brain_age/frozen"
ONH_EMBEDDING_ROOT="${PROJECT_ROOT}/DATA/embeddings"
COHORT_MANIFEST="${PROJECT_ROOT}/preprocessings/FineTunningDataset/brainage_splits/brainage_definitive_subject_manifest.tsv"
ONH_FILELIST="${PROJECT_ROOT}/DATA/filelists/filelist_BrainIAC_raw.txt"
ONH_METADATA="${PROJECT_ROOT}/DATA/all_idps.csv"
OUTPUT_ROOT="${PROJECT_ROOT}/Results/brain_age/onharmony/frozen"
TEST_ROOT="${PROJECT_ROOT}/Results/brain_age/internal_test/frozen"
LOG_ROOT="${PROJECT_ROOT}/Results/brain_age/logs/onharmony_frozen"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

for executable in "${MYLOCAL_PYTHON}" "${NEUROVFM_PYTHON}"; do
    if [[ ! -x "${executable}" ]]; then
        echo "Required Python executable is unavailable: ${executable}" >&2
        exit 1
    fi
done

if command -v klist >/dev/null 2>&1; then
    if [[ -z "${KRB5CCNAME:-}" ]] || ! klist -s; then
        echo "No valid inherited Kerberos cache; refusing an NFS run." >&2
        exit 1
    fi
fi

mkdir -p "${RIDGE_ROOT}" "${OUTPUT_ROOT}" "${TEST_ROOT}" "${LOG_ROOT}"
mkdir -p "/tmp/mpl-frozen-ridge-${USER}"
cd "${PROJECT_ROOT}"

export MPLCONFIGDIR="/tmp/mpl-frozen-ridge-${USER}"
export PYTHONUNBUFFERED=1

declare -A PREFLIGHT_TARGETS=()
PREFLIGHT_COUNT=0

require_file() {
    local label="$1"
    local path="$2"
    if [[ ! -f "${path}" ]]; then
        echo "Missing ${label}: ${path}" >&2
        exit 1
    fi
}

echo "Frozen Ridge fitting, label-blind ON-Harmony prediction, and final test"
echo "  project : ${PROJECT_ROOT}"
echo "  cohort  : ${COHORT_MANIFEST}"
echo "  filelist: ${ONH_FILELIST}"
echo "  metadata: ${ONH_METADATA}"
echo "  ONH     : ${OUTPUT_ROOT}"
echo "  test    : ${TEST_ROOT}"
echo "  order   : fit all -> freeze ON-Harmony all -> evaluate internal_test all"

run_one() {
    local phase="$1"
    local python_executable="$2"
    local model_name="$3"
    local variant="$4"
    local script="$5"
    local embedding_dir="$6"
    local onh_embeddings="$7"
    local fold="${8:-}"
    local ridge_dir="${RIDGE_ROOT}/${model_name}/${variant}"
    local output_dir="${OUTPUT_ROOT}/${model_name}/${variant}"
    local test_dir="${TEST_ROOT}/${model_name}/${variant}"
    local test_receipt="${ridge_dir}/INTERNAL_TEST_EVALUATED.json"
    local log_stem="${model_name}_${variant}_${RUN_ID}"
    local onh_manifest="${onh_embeddings%.csv}_manifest.json"
    local target_key="${model_name}/${variant}"
    local split
    local -a model_args=()

    if [[ -n "${fold}" ]]; then
        model_args=(--fold "${fold}")
    fi

    case "${phase}" in
        preflight)
            if [[ ! -x "${python_executable}" ]]; then
                echo "Python executable is unavailable: ${python_executable}" >&2
                exit 1
            fi
            if [[ -n "${PREFLIGHT_TARGETS[${target_key}]:-}" ]]; then
                echo "Duplicate frozen representation in launcher: ${target_key}" >&2
                exit 1
            fi
            PREFLIGHT_TARGETS["${target_key}"]=1

            require_file "Ridge entry point" "${PROJECT_ROOT}/${script}"
            for split in train validation internal_test; do
                require_file "${split} embeddings" "${embedding_dir}/${split}.tsv"
                require_file "${split} embedding audit" \
                    "${embedding_dir}/${split}.tsv.audit.json"
            done
            require_file "ON-Harmony embeddings" "${onh_embeddings}"
            require_file "ON-Harmony extractor manifest" "${onh_manifest}"

            if [[ -e "${ridge_dir}" ]]; then
                echo "Ridge output already exists; refusing to start: ${ridge_dir}" >&2
                exit 1
            fi
            if [[ -e "${output_dir}" ]]; then
                echo "ON-Harmony output already exists; refusing to start: ${output_dir}" >&2
                exit 1
            fi
            if [[ -e "${test_dir}" || -e "${test_receipt}" ]]; then
                echo "Internal-test output or receipt already exists; refusing to start: ${target_key}" >&2
                exit 1
            fi

            echo "[$(date --iso-8601=seconds)] AUDIT SOURCE ${target_key}"
            "${python_executable}" -u "${script}" audit-onharmony \
                "${model_args[@]}" \
                --embeddings "${onh_embeddings}" \
                --embedding-manifest "${onh_manifest}" \
                --filelist "${ONH_FILELIST}" \
                --metadata "${ONH_METADATA}" \
                2>&1 | tee "${LOG_ROOT}/${log_stem}_source_audit.log"
            PREFLIGHT_COUNT=$((PREFLIGHT_COUNT + 1))
            ;;
        fit)
            if [[ -e "${ridge_dir}" ]]; then
                echo "Ridge output already exists; refusing to reuse or overwrite it: ${ridge_dir}" >&2
                exit 1
            fi
            echo "[$(date --iso-8601=seconds)] FIT ${model_name}/${variant}"
            "${python_executable}" -u "${script}" fit \
                "${model_args[@]}" \
                --manifest "${COHORT_MANIFEST}" \
                --train-embeddings "${embedding_dir}/train.tsv" \
                --validation-embeddings "${embedding_dir}/validation.tsv" \
                --output-dir "${ridge_dir}" \
                2>&1 | tee "${LOG_ROOT}/${log_stem}_fit.log"
            ;;
        onharmony)
            if [[ -e "${output_dir}" ]]; then
                echo "ON-Harmony output already exists; refusing to reuse or overwrite it: ${output_dir}" >&2
                exit 1
            fi
            echo "[$(date --iso-8601=seconds)] PREDICT ${model_name}/${variant}"
            "${python_executable}" -u "${script}" predict-onharmony \
                "${model_args[@]}" \
                --run-dir "${ridge_dir}" \
                --embeddings "${onh_embeddings}" \
                --embedding-manifest "${onh_manifest}" \
                --filelist "${ONH_FILELIST}" \
                --metadata "${ONH_METADATA}" \
                --output-dir "${output_dir}" \
                2>&1 | tee "${LOG_ROOT}/${log_stem}_onharmony.log"
            ;;
        test)
            if [[ -e "${test_dir}" || -e "${test_receipt}" ]]; then
                echo "Internal-test output or receipt already exists; refusing to reuse it: ${model_name}/${variant}" >&2
                exit 1
            fi
            echo "[$(date --iso-8601=seconds)] TEST ${model_name}/${variant}"
            "${python_executable}" -u "${script}" evaluate \
                "${model_args[@]}" \
                --manifest "${COHORT_MANIFEST}" \
                --run-dir "${ridge_dir}" \
                --test-embeddings "${embedding_dir}/internal_test.tsv" \
                --output-dir "${test_dir}" \
                --acknowledge-locked-test \
                2>&1 | tee "${LOG_ROOT}/${log_stem}_internal_test.log"
            ;;
        *)
            echo "Unknown launcher phase: ${phase}" >&2
            exit 1
            ;;
    esac
}

run_matrix() {
    local phase="$1"
    local fold

    run_one "${phase}" "${MYLOCAL_PYTHON}" "3DNeuroSimCLR" "canonical" \
        "FineTuning/3DNeuroSimCLR/fit_ridge.py" \
        "${EMBEDDING_ROOT}/3DNeuroSimCLR/canonical" \
        "${ONH_EMBEDDING_ROOT}/embeddings_3DNeuroSimCLR.csv"

    for fold in 0 1 2 3 4; do
        run_one "${phase}" "${MYLOCAL_PYTHON}" "AnatCL" "fold${fold}" \
            "FineTuning/AnatCL/fit_ridge.py" \
            "${EMBEDDING_ROOT}/AnatCL/fold${fold}" \
            "${ONH_EMBEDDING_ROOT}/embeddings_AnatCL_fold${fold}.csv" \
            "${fold}"
    done

    run_one "${phase}" "${MYLOCAL_PYTHON}" "BrainIAC" "canonical" \
        "FineTuning/BrainIAC/fit_ridge.py" \
        "${EMBEDDING_ROOT}/BrainIAC/canonical" \
        "${ONH_EMBEDDING_ROOT}/embeddings_BrainIAC.csv"

    run_one "${phase}" "${MYLOCAL_PYTHON}" "BrainSegFounder" "bottleneck_mean" \
        "FineTuning/BrainSegFounder/fit_ridge.py" \
        "${EMBEDDING_ROOT}/BrainSegFounder/canonical" \
        "${ONH_EMBEDDING_ROOT}/embeddings_BrainSegFounder.csv"

    run_one "${phase}" "${MYLOCAL_PYTHON}" "Curia2" "canonical" \
        "FineTuning/Curia2/fit_ridge.py" \
        "${EMBEDDING_ROOT}/Curia2/canonical" \
        "${ONH_EMBEDDING_ROOT}/embeddings_Curia2.csv"

    run_one "${phase}" "${NEUROVFM_PYTHON}" "NeuroVFM" "canonical" \
        "FineTuning/NeuroVFM/fit_ridge.py" \
        "${EMBEDDING_ROOT}/NeuroVFM/canonical" \
        "${ONH_EMBEDDING_ROOT}/embeddings_NeuroVFM.csv"

    run_one "${phase}" "${MYLOCAL_PYTHON}" "yAware" "canonical" \
        "FineTuning/yAware/fit_ridge.py" \
        "${EMBEDDING_ROOT}/yAware/canonical" \
        "${ONH_EMBEDDING_ROOT}/embeddings_yAware.csv"
}

require_file "definitive cohort manifest" "${COHORT_MANIFEST}"
require_file "ON-Harmony filelist" "${ONH_FILELIST}"
require_file "ON-Harmony metadata" "${ONH_METADATA}"

echo "[$(date --iso-8601=seconds)] PREFLIGHT: validating all frozen inputs and targets"
run_matrix preflight
if [[ "${PREFLIGHT_COUNT}" -ne 11 ]]; then
    echo "Internal launcher error: expected 11 representations, found ${PREFLIGHT_COUNT}" >&2
    exit 1
fi
echo "[$(date --iso-8601=seconds)] PREFLIGHT PASS: ${PREFLIGHT_COUNT} fresh representations"

run_matrix fit
echo "[$(date --iso-8601=seconds)] LOCKED: all 11 train/validation Ridge heads"
run_matrix onharmony
echo "[$(date --iso-8601=seconds)] FROZEN: all 11 label-blind ON-Harmony predictions"
run_matrix test
echo "[$(date --iso-8601=seconds)] COMPLETE: 11 frozen Ridge representations"
