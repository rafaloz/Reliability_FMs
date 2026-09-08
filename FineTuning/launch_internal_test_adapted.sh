#!/usr/bin/env bash
# Evaluate every retained adapted checkpoint on the locked internal test split.

set -euo pipefail

GPU="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MYLOCAL_PYTHON="${MYLOCAL_PYTHON:-/home/export/personal/rnavgon/.conda/envs/myLocalEnv/bin/python}"
NEUROVFM_PYTHON="${NEUROVFM_PYTHON:-/home/export/personal/rnavgon/.conda/envs/neurovfm/bin/python}"
CURIA2_MODEL_DIR="${CURIA2_MODEL_DIR:-/datos/work/rnavgon/FoundationModels_Paper/Models/Curia-2}"
MANIFEST="${PROJECT_ROOT}/preprocessings/FineTunningDataset/brainage_splits/brainage_definitive_subject_manifest.tsv"
ADAPTED_ROOT="${PROJECT_ROOT}/Results/brain_age/adapted"
ONHARMONY_ROOT="${PROJECT_ROOT}/Results/brain_age/onharmony/adapted"
OUTPUT_ROOT="${PROJECT_ROOT}/Results/brain_age/internal_test/adapted"
LOG_ROOT="${PROJECT_ROOT}/Results/brain_age/logs/internal_test_adapted"
NEUROVFM_CACHE="${PROJECT_ROOT}/preprocessings/FineTunningDataset/preproc_NeuroVFM"
NEUROSIMCLR_CHECKPOINT="${PROJECT_ROOT}/Models/3D-Neuro-SimCLR/simclr_3d_brain_foundation.tar"
NEUROSIMCLR_PREPROC="${PROJECT_ROOT}/preprocessings/FineTunningDataset/preproc_3DNeuroSimCLR"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

NEUROSIMCLR_RUN_SPECS=(
    "selection_seed0_intermediate:0"
    "final_seed1_intermediate:1"
    "final_seed2_intermediate:2"
)
ANATCL_RUN_SPECS=(
    "fold0_seed0_high:fold0:0"
    "fold0_seed1_high:fold0:1"
    "fold0_seed2_high:fold0:2"
    "fold1_seed0_high:fold1:0"
    "fold2_seed0_high:fold2:0"
    "fold3_seed0_high:fold3:0"
    "fold4_seed0_high:fold4:0"
)
HIGH_RUN_SPECS=(
    "selection_seed0_high:0"
    "final_seed1_high:1"
    "final_seed2_high:2"
)

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
mkdir -p "/tmp/mpl-internal-test-${USER}" "/tmp/triton-neurovfm-${USER}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR="/tmp/mpl-internal-test-${USER}"
export TRITON_CACHE_DIR="/tmp/triton-neurovfm-${USER}"
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

require_directory() {
    local label="$1"
    local path="$2"
    if [[ ! -d "${path}" ]]; then
        echo "Missing ${label}: ${path}" >&2
        exit 1
    fi
}

preflight_run() {
    local family="$1"
    local expected_model="$2"
    local expected_variant="$3"
    local run_name="$4"
    local selection_name="$5"
    local expected_seed="$6"
    local registration_mode="${7:-register}"
    local expected_selection_variant="${expected_variant}"
    local target_key="${family}/${run_name}"

    if [[ "${family}" == "AnatCL" ]]; then
        expected_selection_variant="fold0"
    fi
    if [[ "${registration_mode}" == "register" ]]; then
        if [[ -n "${PREFLIGHT_TARGETS[${target_key}]:-}" ]]; then
            echo "Duplicate adapted evaluation target: ${target_key}" >&2
            exit 1
        fi
        PREFLIGHT_TARGETS["${target_key}"]="${expected_model}|${expected_variant}|${expected_seed}"
        PREFLIGHT_COUNT=$((PREFLIGHT_COUNT + 1))
    elif [[ "${registration_mode}" != "verify" ]]; then
        echo "Unknown preflight registration mode: ${registration_mode}" >&2
        exit 1
    fi

    "${MYLOCAL_PYTHON}" - \
        "${ADAPTED_ROOT}/${family}/${run_name}" \
        "${ADAPTED_ROOT}/${family}/${selection_name}" \
        "${ONHARMONY_ROOT}/${family}/${run_name}" \
        "${OUTPUT_ROOT}/${family}/${run_name}" \
        "${expected_model}" "${expected_variant}" "${expected_seed}" \
        "${expected_selection_variant}" "${MANIFEST}" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path


run_dir, selection_dir, onh_dir, output_dir = map(Path, sys.argv[1:5])
expected_model, expected_variant = sys.argv[5], sys.argv[6]
expected_seed, expected_selection_variant = int(sys.argv[7]), sys.argv[8]
manifest_path = Path(sys.argv[9])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


lock_path = run_dir / "LOCKED.json"
checkpoint_path = run_dir / "selected_model.pt"
selection_path = selection_dir / "SELECTED_LR.json"
lock = read_json(lock_path)
selection = read_json(selection_path)
expected_lock = {
    "status": "LOCKED_AFTER_TRAIN_VALIDATION",
    "arm": "adapted",
    "model": expected_model,
    "variant": expected_variant,
    "internal_test_accessed": False,
    "on_harmony_accessed": False,
}
bad_lock = {
    key: {"expected": value, "observed": lock.get(key)}
    for key, value in expected_lock.items()
    if lock.get(key) != value
}
if bad_lock:
    raise ValueError(f"Invalid retained lock {lock_path}: {bad_lock}")
if not checkpoint_path.is_file() or sha256(checkpoint_path) != lock.get("checkpoint_sha256"):
    raise ValueError(f"Checkpoint hash mismatch: {checkpoint_path}")

arguments = lock.get("run_config", {}).get("arguments", {})
if int(arguments.get("seed")) != expected_seed:
    raise ValueError(f"Unexpected training seed in {lock_path}")
lock_manifest_sha256 = lock.get("manifest", {}).get("manifest_sha256")
if not isinstance(lock_manifest_sha256, str) or not lock_manifest_sha256:
    raise ValueError(f"Training lock lacks a manifest SHA-256: {lock_path}")
if not manifest_path.is_file() or sha256(manifest_path) != lock_manifest_sha256:
    raise ValueError(
        f"Definitive cohort manifest differs from the training lock: {manifest_path}"
    )
expected_selection = {
    "status": "LR_SELECTED_AFTER_TRAIN_VALIDATION",
    "model": expected_model,
    "variant": expected_selection_variant,
    "manifest_sha256": lock_manifest_sha256,
    "selection_seed": 0,
    "internal_test_accessed": False,
    "on_harmony_accessed": False,
}
bad_selection = {
    key: {"expected": value, "observed": selection.get(key)}
    for key, value in expected_selection.items()
    if selection.get(key) != value
}
if bad_selection:
    raise ValueError(f"Invalid validation-only LR selection {selection_path}: {bad_selection}")
if selection.get("selected_lr_config") != arguments.get("lr_config"):
    raise ValueError(f"Run does not use the validation-selected LR: {run_dir}")

selected_run_dir = Path(str(selection.get("selected_run_dir", ""))).resolve()
selected_lock_path = selected_run_dir / "LOCKED.json"
selected_checkpoint_path = selected_run_dir / "selected_model.pt"
selected_lock = read_json(selected_lock_path)
if not selected_checkpoint_path.is_file():
    raise FileNotFoundError(selected_checkpoint_path)
expected_selected_lock = {
    "status": "LOCKED_AFTER_TRAIN_VALIDATION",
    "arm": "adapted",
    "model": expected_model,
    "variant": expected_selection_variant,
    "internal_test_accessed": False,
    "on_harmony_accessed": False,
}
bad_selected_lock = {
    key: {"expected": value, "observed": selected_lock.get(key)}
    for key, value in expected_selected_lock.items()
    if selected_lock.get(key) != value
}
if bad_selected_lock:
    raise ValueError(f"Invalid LR-selection anchor {selected_lock_path}: {bad_selected_lock}")
if selected_lock.get("manifest", {}).get("manifest_sha256") != lock_manifest_sha256:
    raise ValueError(f"Selection anchor uses a different manifest: {selected_lock_path}")
selected_checkpoint_sha256 = sha256(selected_checkpoint_path)
if (
    selected_checkpoint_sha256 != selection.get("selected_checkpoint_sha256")
    or selected_checkpoint_sha256 != selected_lock.get("checkpoint_sha256")
):
    raise ValueError(f"Selection checkpoint hash mismatch: {selected_checkpoint_path}")
if expected_seed == 0 and expected_variant == expected_selection_variant:
    if selection.get("selected_checkpoint_sha256") != lock.get("checkpoint_sha256"):
        raise ValueError(f"Selection-seed checkpoint is not the selected checkpoint: {run_dir}")

onh_predictions = onh_dir / "predictions.tsv"
onh_audit_path = onh_dir / "prediction_audit.json"
onh_audit = read_json(onh_audit_path)
with onh_predictions.open(encoding="utf-8", newline="") as handle:
    onh_rows = sum(1 for _ in csv.DictReader(handle, delimiter="\t"))
expected_onh = {
    "status": "ON_HARMONY_PREDICTED",
    "arm": "adapted",
    "model": expected_model,
    "variant": expected_variant,
    "label_blind": True,
    "outcome_labels_loaded": False,
    "n_predictions": 165,
    "training_seed": expected_seed,
    "adapted_checkpoint_sha256": lock["checkpoint_sha256"],
}
bad_onh = {
    key: {"expected": value, "observed": onh_audit.get(key)}
    for key, value in expected_onh.items()
    if onh_audit.get(key) != value
}
if bad_onh or onh_rows != 165:
    raise ValueError(
        f"ON-Harmony prerequisite failed for {run_dir}: audit={bad_onh}, rows={onh_rows}"
    )
if onh_audit.get("predictions_artifact_sha256") != sha256(onh_predictions):
    raise ValueError(f"ON-Harmony predictions hash mismatch: {onh_predictions}")

predictions_path = output_dir / "predictions_internal_test.tsv"
metrics_path = output_dir / "metrics_internal_test.json"
receipt_path = run_dir / "INTERNAL_TEST_EVALUATED.json"
state = [predictions_path.exists(), metrics_path.exists(), receipt_path.exists()]
if any(state) and not all(state):
    raise ValueError(
        f"Partial internal-test state requires manual audit before continuing: "
        f"output={output_dir}, receipt={receipt_path}"
    )
if all(state):
    metrics = read_json(metrics_path)
    receipt = read_json(receipt_path)
    with predictions_path.open(encoding="utf-8", newline="") as handle:
        test_rows = sum(1 for _ in csv.DictReader(handle, delimiter="\t"))
    expected_metrics = {
        "arm": "adapted",
        "model": expected_model,
        "variant": expected_variant,
        "checkpoint_sha256": lock["checkpoint_sha256"],
        "internal_test_subjects": 140,
        "internal_test_accessed": True,
        "on_harmony_accessed": False,
    }
    bad_metrics = {
        key: {"expected": value, "observed": metrics.get(key)}
        for key, value in expected_metrics.items()
        if metrics.get(key) != value
    }
    expected_receipt = {
        "status": "INTERNAL_TEST_EVALUATED",
        "arm": "adapted",
        "model": expected_model,
        "variant": expected_variant,
        "internal_test_accessed": True,
    }
    bad_receipt = {
        key: {"expected": value, "observed": receipt.get(key)}
        for key, value in expected_receipt.items()
        if receipt.get(key) != value
    }
    expected_selection_provenance = {
        "selection_artifact_sha256": sha256(selection_path),
        "selected_lr_config": selection["selected_lr_config"],
        "selection_model": expected_model,
        "selection_variant": expected_selection_variant,
        "selection_seed": 0,
        "selection_checkpoint_sha256": selected_checkpoint_sha256,
        "evaluation_run": str(run_dir.resolve()),
        "evaluation_variant": expected_variant,
        "evaluation_seed": expected_seed,
        "cross_variant_lr_transfer": expected_variant != expected_selection_variant,
    }
    observed_selection_provenance = metrics.get("lr_selection", {})
    bad_selection_provenance = {
        key: {"expected": value, "observed": observed_selection_provenance.get(key)}
        for key, value in expected_selection_provenance.items()
        if observed_selection_provenance.get(key) != value
    }
    if bad_metrics or bad_receipt or bad_selection_provenance:
        raise ValueError(
            f"Completed internal test is bound to the wrong run: "
            f"metrics={bad_metrics}, receipt={bad_receipt}, "
            f"selection={bad_selection_provenance}"
        )
    if receipt.get("lr_selection") != observed_selection_provenance:
        raise ValueError(f"Metrics and receipt have different LR provenance: {output_dir}")
    if metrics.get("manifest", {}).get("manifest_sha256") != lock_manifest_sha256:
        raise ValueError(f"Completed internal test uses a different manifest: {output_dir}")
    if test_rows != 140 or metrics.get("metrics", {}).get("n") != 140:
        raise ValueError(f"Completed internal test has the wrong row count: {output_dir}")
    if metrics.get("predictions_artifact_sha256") != sha256(predictions_path):
        raise ValueError(f"Internal-test metrics do not bind predictions: {metrics_path}")
    if receipt.get("result_sha256") != sha256(metrics_path):
        raise ValueError(f"Internal-test metrics hash mismatch: {metrics_path}")
    if receipt.get("predictions_artifact_sha256") != sha256(predictions_path):
        raise ValueError(f"Internal-test predictions hash mismatch: {predictions_path}")
    print(f"COMPLETE {expected_model}/{expected_variant} seed={expected_seed}: {output_dir}")
else:
    if output_dir.exists():
        raise ValueError(f"Unexpected empty or partial output directory: {output_dir}")
    print(f"READY    {expected_model}/{expected_variant} seed={expected_seed}: {run_dir}")
PY
}

preflight_all() {
    local run_spec run_name variant seed

    for run_spec in "${NEUROSIMCLR_RUN_SPECS[@]}"; do
        run_name="${run_spec%%:*}"
        seed="${run_spec##*:}"
        preflight_run 3DNeuroSimCLR 3D-Neuro-SimCLR canonical \
            "${run_name}" lr_selection_seed0 "${seed}"
    done

    for run_spec in "${ANATCL_RUN_SPECS[@]}"; do
        IFS=: read -r run_name variant seed <<<"${run_spec}"
        preflight_run AnatCL AnatCL "${variant}" \
            "${run_name}" fold0_lr_selection "${seed}"
    done

    for run_spec in "${HIGH_RUN_SPECS[@]}"; do
        run_name="${run_spec%%:*}"
        seed="${run_spec##*:}"
        preflight_run BrainIAC BrainIAC canonical \
            "${run_name}" lr_selection_seed0 "${seed}"
        preflight_run BrainSegFounder BrainSegFounder bottleneck_mean \
            "${run_name}" lr_selection_seed0 "${seed}"
        preflight_run Curia2 Curia-2 canonical \
            "${run_name}" lr_selection_seed0 "${seed}"
        preflight_run NeuroVFM NeuroVFM canonical \
            "${run_name}" lr_selection_seed0 "${seed}"
        preflight_run yAware y-Aware canonical \
            "${run_name}" lr_selection_seed0 "${seed}"
    done
}

run_and_verify() {
    local family="$1"
    local expected_model="$2"
    local expected_variant="$3"
    local run_name="$4"
    local selection_name="$5"
    local expected_seed="$6"
    shift 6

    local output_dir="${OUTPUT_ROOT}/${family}/${run_name}"
    local receipt="${ADAPTED_ROOT}/${family}/${run_name}/INTERNAL_TEST_EVALUATED.json"
    local target_key="${family}/${run_name}"
    if [[ -z "${PREFLIGHT_TARGETS[${target_key}]:-}" ]]; then
        echo "Evaluation target was not registered by the global preflight: ${target_key}" >&2
        exit 1
    fi
    if [[ -f "${output_dir}/predictions_internal_test.tsv" \
        && -f "${output_dir}/metrics_internal_test.json" \
        && -f "${receipt}" ]]; then
        echo "[$(date --iso-8601=seconds)] SKIP complete ${family}/${run_name}"
        return
    fi

    mkdir -p "${OUTPUT_ROOT}/${family}"
    local log_path="${LOG_ROOT}/${family}_${run_name}_${RUN_ID}.log"
    echo "[$(date --iso-8601=seconds)] START ${family}/${run_name}"
    "$@" 2>&1 | tee "${log_path}"
    preflight_run "${family}" "${expected_model}" "${expected_variant}" \
        "${run_name}" "${selection_name}" "${expected_seed}" verify
    echo "[$(date --iso-8601=seconds)] DONE  ${family}/${run_name}"
}

echo "Adapted internal-test evaluation"
echo "  project       : ${PROJECT_ROOT}"
echo "  physical GPU  : ${GPU}"
echo "  Kerberos cache: ${KRB5CCNAME:-not exported}"
echo "  checkpoints   : 25 (7 model families)"
echo "  test subjects : 140"
echo "  outputs       : ${OUTPUT_ROOT}"
require_file "definitive cohort manifest" "${MANIFEST}"
require_file "3D-Neuro-SimCLR checkpoint" "${NEUROSIMCLR_CHECKPOINT}"
require_directory "3D-Neuro-SimCLR preprocessing root" "${NEUROSIMCLR_PREPROC}"
echo "[$(date --iso-8601=seconds)] PRE-FLIGHT all retained checkpoints"
preflight_all
if [[ "${PREFLIGHT_COUNT}" -ne 25 ]]; then
    echo "Internal launcher error: expected 25 unique targets, found ${PREFLIGHT_COUNT}" >&2
    exit 1
fi

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "Pre-flight complete; internal test was not evaluated."
    exit 0
fi
if [[ "${ACKNOWLEDGE_LOCKED_TEST:-}" != "YES" ]]; then
    echo "Set ACKNOWLEDGE_LOCKED_TEST=YES to authorize the one-time evaluation." >&2
    exit 1
fi

echo "[$(date --iso-8601=seconds)] CACHE NeuroVFM internal-test inputs"
"${NEUROVFM_PYTHON}" -u FineTuning/NeuroVFM/extract_embeddings.py cache \
    --manifest "${MANIFEST}" \
    --cache-root "${NEUROVFM_CACHE}" \
    --splits internal_test \
    --acknowledge-locked-test \
    2>&1 | tee "${LOG_ROOT}/NeuroVFM_internal_test_cache_${RUN_ID}.log"

for run_spec in "${NEUROSIMCLR_RUN_SPECS[@]}"; do
    run_name="${run_spec%%:*}"
    seed="${run_spec##*:}"
    run_and_verify 3DNeuroSimCLR 3D-Neuro-SimCLR canonical "${run_name}" lr_selection_seed0 "${seed}" \
        "${MYLOCAL_PYTHON}" -u FineTuning/3DNeuroSimCLR/finetune.py evaluate \
        --manifest "${MANIFEST}" \
        --checkpoint "${NEUROSIMCLR_CHECKPOINT}" \
        --preproc-root "${NEUROSIMCLR_PREPROC}" \
        --device cuda --amp bfloat16 --seed "${seed}" \
        --run-dir "${ADAPTED_ROOT}/3DNeuroSimCLR/${run_name}" \
        --selection-dir "${ADAPTED_ROOT}/3DNeuroSimCLR/lr_selection_seed0" \
        --output-dir "${OUTPUT_ROOT}/3DNeuroSimCLR/${run_name}" \
        --acknowledge-locked-test
done

for run_spec in "${ANATCL_RUN_SPECS[@]}"; do
    IFS=: read -r run_name variant seed <<<"${run_spec}"
    fold="${variant#fold}"
    run_and_verify AnatCL AnatCL "${variant}" "${run_name}" fold0_lr_selection "${seed}" \
        "${MYLOCAL_PYTHON}" -u FineTuning/AnatCL/finetune.py evaluate \
        --manifest "${MANIFEST}" --fold "${fold}" \
        --checkpoint-dir Models/AnatCL/NEW_anatcl-g3 \
        --preproc-root preprocessings/FineTunningDataset/preproc_AnatCL \
        --device cuda --amp bfloat16 --seed "${seed}" \
        --run-dir "${ADAPTED_ROOT}/AnatCL/${run_name}" \
        --selection-dir "${ADAPTED_ROOT}/AnatCL/fold0_lr_selection" \
        --output-dir "${OUTPUT_ROOT}/AnatCL/${run_name}" \
        --acknowledge-locked-test
done

for run_spec in "${HIGH_RUN_SPECS[@]}"; do
    run_name="${run_spec%%:*}"
    seed="${run_spec##*:}"
    run_and_verify BrainIAC BrainIAC canonical "${run_name}" lr_selection_seed0 "${seed}" \
        "${MYLOCAL_PYTHON}" -u FineTuning/BrainIAC/finetune.py evaluate \
        --manifest "${MANIFEST}" --checkpoint Models/BrainIAC/BrainIAC.ckpt \
        --preproc-root preprocessings/FineTunningDataset/preproc_BrainIAC \
        --device cuda --amp bfloat16 --seed "${seed}" \
        --run-dir "${ADAPTED_ROOT}/BrainIAC/${run_name}" \
        --selection-dir "${ADAPTED_ROOT}/BrainIAC/lr_selection_seed0" \
        --output-dir "${OUTPUT_ROOT}/BrainIAC/${run_name}" \
        --acknowledge-locked-test
done

for run_spec in "${HIGH_RUN_SPECS[@]}"; do
    run_name="${run_spec%%:*}"
    seed="${run_spec##*:}"
    run_and_verify BrainSegFounder BrainSegFounder bottleneck_mean "${run_name}" lr_selection_seed0 "${seed}" \
        "${MYLOCAL_PYTHON}" -u FineTuning/BrainSegFounder/finetune.py evaluate \
        --manifest "${MANIFEST}" \
        --checkpoint Models/BrainSegFounder/64-gpu-model_bestValRMSE.pt \
        --preproc-root preprocessings/FineTunningDataset/preproc_BrainSegFounder \
        --device cuda --amp bfloat16 --seed "${seed}" \
        --run-dir "${ADAPTED_ROOT}/BrainSegFounder/${run_name}" \
        --selection-dir "${ADAPTED_ROOT}/BrainSegFounder/lr_selection_seed0" \
        --output-dir "${OUTPUT_ROOT}/BrainSegFounder/${run_name}" \
        --acknowledge-locked-test
done

for run_spec in "${HIGH_RUN_SPECS[@]}"; do
    run_name="${run_spec%%:*}"
    seed="${run_spec##*:}"
    run_and_verify Curia2 Curia-2 canonical "${run_name}" lr_selection_seed0 "${seed}" \
        "${MYLOCAL_PYTHON}" -u FineTuning/Curia2/finetune.py evaluate \
        --manifest "${MANIFEST}" --model-dir "${CURIA2_MODEL_DIR}" \
        --slice-batch-size 192 --device cuda --amp bfloat16 --seed "${seed}" \
        --run-dir "${ADAPTED_ROOT}/Curia2/${run_name}" \
        --selection-dir "${ADAPTED_ROOT}/Curia2/lr_selection_seed0" \
        --output-dir "${OUTPUT_ROOT}/Curia2/${run_name}" \
        --acknowledge-locked-test
done

for run_spec in "${HIGH_RUN_SPECS[@]}"; do
    run_name="${run_spec%%:*}"
    seed="${run_spec##*:}"
    run_and_verify NeuroVFM NeuroVFM canonical "${run_name}" lr_selection_seed0 "${seed}" \
        "${NEUROVFM_PYTHON}" -u FineTuning/NeuroVFM/finetune.py evaluate \
        --manifest "${MANIFEST}" --model-source mlinslab/neurovfm-encoder \
        --model-revision d5194fc70a162185f8ef062e362bd522a35312a9 \
        --cache-root "${NEUROVFM_CACHE}" --device cuda --amp bfloat16 --seed "${seed}" \
        --run-dir "${ADAPTED_ROOT}/NeuroVFM/${run_name}" \
        --selection-dir "${ADAPTED_ROOT}/NeuroVFM/lr_selection_seed0" \
        --output-dir "${OUTPUT_ROOT}/NeuroVFM/${run_name}" \
        --acknowledge-locked-test
done

for run_spec in "${HIGH_RUN_SPECS[@]}"; do
    run_name="${run_spec%%:*}"
    seed="${run_spec##*:}"
    run_and_verify yAware y-Aware canonical "${run_name}" lr_selection_seed0 "${seed}" \
        "${MYLOCAL_PYTHON}" -u FineTuning/yAware/finetune.py evaluate \
        --manifest "${MANIFEST}" \
        --checkpoint Models/y-Aware/DenseNet121_BHB-10K_yAwareContrastive.pth \
        --preproc-root preprocessings/FineTunningDataset/preproc_yAware \
        --qc-root preprocessings/FineTunningDataset/preproc_AnatCL \
        --device cuda --amp bfloat16 --seed "${seed}" \
        --run-dir "${ADAPTED_ROOT}/yAware/${run_name}" \
        --selection-dir "${ADAPTED_ROOT}/yAware/lr_selection_seed0" \
        --output-dir "${OUTPUT_ROOT}/yAware/${run_name}" \
        --acknowledge-locked-test
done

SUMMARY_TARGET_SPECS=()
for target_key in "${!PREFLIGHT_TARGETS[@]}"; do
    SUMMARY_TARGET_SPECS+=(
        "${target_key}|${PREFLIGHT_TARGETS[${target_key}]}"
    )
done

"${MYLOCAL_PYTHON}" - \
    "${OUTPUT_ROOT}" "${PROJECT_ROOT}" "${MANIFEST}" \
    "${SCRIPT_DIR}/launch_internal_test_adapted.sh" \
    "${SUMMARY_TARGET_SPECS[@]}" <<'PY'
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
from pathlib import Path


root = Path(sys.argv[1])
project_root = Path(sys.argv[2]).resolve()
manifest_path = Path(sys.argv[3])
launcher_path = Path(sys.argv[4])
target_specs = sys.argv[5:]
hash_pattern = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant {value!r} in {path}")

    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or hash_pattern.fullmatch(value) is None:
        raise ValueError(f"Invalid SHA-256 for {label}: {value!r}")
    return value


def project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def immutable_publish(path: Path, content: bytes) -> str:
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise FileExistsError(
                f"Immutable artifact exists with different content: {path}"
            )
        return "VERIFIED_UNCHANGED"

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != content:
                raise FileExistsError(
                    f"Immutable artifact was concurrently published with "
                    f"different content: {path}"
                )
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


expected = {}
for spec in target_specs:
    parts = spec.split("|")
    if len(parts) != 4:
        raise ValueError(f"Malformed internal-test target specification: {spec!r}")
    target, model, variant, seed_text = parts
    try:
        family, run = target.split("/", 1)
    except ValueError as error:
        raise ValueError(
            f"Malformed internal-test target key: {target!r}"
        ) from error
    if target in expected:
        raise ValueError(f"Duplicate internal-test target specification: {target}")
    expected[target] = {
        "family": family,
        "run": run,
        "model": model,
        "variant": variant,
        "seed": int(seed_text),
    }
if len(expected) != 25:
    raise RuntimeError(f"Expected 25 registered targets, found {len(expected)}")

observed_paths = sorted(root.glob("*/*/metrics_internal_test.json"))
observed = {
    f"{path.relative_to(root).parts[0]}/{path.relative_to(root).parts[1]}": path
    for path in observed_paths
}
if len(observed) != len(observed_paths):
    raise ValueError("Multiple metrics artifacts resolve to the same target")
missing = sorted(set(expected) - set(observed))
unexpected = sorted(set(observed) - set(expected))
if missing or unexpected:
    raise RuntimeError(
        f"Internal-test result matrix mismatch: missing={missing}, "
        f"unexpected={unexpected}"
    )

manifest_sha256 = sha256(manifest_path)
rows = []
sources = []
for target in sorted(expected):
    identity = expected[target]
    path = observed[target]
    payload = read_json(path)
    metrics = payload["metrics"]
    selection = payload.get("lr_selection", {})
    predictions_path = path.parent / "predictions_internal_test.tsv"

    expected_payload = {
        "arm": "adapted",
        "model": identity["model"],
        "variant": identity["variant"],
        "internal_test_subjects": 140,
        "internal_test_accessed": True,
        "on_harmony_accessed": False,
    }
    bad_payload = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected_payload.items()
        if payload.get(key) != value
    }
    if bad_payload:
        raise ValueError(f"Wrong result identity for {target}: {bad_payload}")
    if payload.get("manifest", {}).get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"Result uses a different cohort manifest: {path}")
    if not isinstance(selection, dict):
        raise TypeError(f"Invalid LR-selection provenance: {path}")
    expected_selection = {
        "evaluation_variant": identity["variant"],
        "evaluation_seed": identity["seed"],
    }
    bad_selection = {
        key: {"expected": value, "observed": selection.get(key)}
        for key, value in expected_selection.items()
        if selection.get(key) != value
    }
    if bad_selection:
        raise ValueError(f"Wrong evaluation provenance for {target}: {bad_selection}")
    if not isinstance(metrics, dict) or metrics.get("n") != 140:
        raise ValueError(f"Wrong metric count for {target}: {metrics!r}")
    for metric_name in ("mae_years", "rmse_years", "r2", "mean_error_years"):
        value = metrics.get(metric_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Non-numeric {metric_name} for {target}: {value!r}")
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite {metric_name} for {target}: {value!r}")

    checkpoint_sha256 = require_sha256(
        payload.get("checkpoint_sha256"), f"{target} checkpoint"
    )
    selection_artifact_sha256 = require_sha256(
        selection.get("selection_artifact_sha256"),
        f"{target} LR-selection artifact",
    )
    selection_checkpoint_sha256 = require_sha256(
        selection.get("selection_checkpoint_sha256"),
        f"{target} LR-selection checkpoint",
    )
    predictions_sha256 = require_sha256(
        payload.get("predictions_artifact_sha256"),
        f"{target} predictions",
    )
    if not predictions_path.is_file() or sha256(predictions_path) != predictions_sha256:
        raise ValueError(f"Predictions hash mismatch for {target}: {predictions_path}")
    with predictions_path.open(encoding="utf-8", newline="") as handle:
        prediction_rows = sum(1 for _ in csv.DictReader(handle, delimiter="\t"))
    if prediction_rows != 140:
        raise ValueError(
            f"Expected 140 prediction rows for {target}, found {prediction_rows}"
        )

    rows.append(
        {
            "model": payload["model"],
            "variant": payload["variant"],
            "run": identity["run"],
            "seed": identity["seed"],
            "n": metrics["n"],
            "mae_years": metrics["mae_years"],
            "rmse_years": metrics["rmse_years"],
            "r2": metrics["r2"],
            "mean_error_years": metrics["mean_error_years"],
            "checkpoint_sha256": checkpoint_sha256,
        }
    )
    sources.append(
        {
            "target": target,
            "model": identity["model"],
            "variant": identity["variant"],
            "run": identity["run"],
            "seed": identity["seed"],
            "metrics_artifact": project_path(path),
            "metrics_artifact_sha256": sha256(path),
            "predictions_artifact": project_path(predictions_path),
            "predictions_artifact_sha256": predictions_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "selection_artifact": selection.get("selection_artifact"),
            "selection_artifact_sha256": selection_artifact_sha256,
            "selection_checkpoint_sha256": selection_checkpoint_sha256,
        }
    )

fieldnames = [
    "model",
    "variant",
    "run",
    "seed",
    "n",
    "mae_years",
    "rmse_years",
    "r2",
    "mean_error_years",
    "checkpoint_sha256",
]
buffer = io.StringIO(newline="")
writer = csv.DictWriter(
    buffer, fieldnames=fieldnames, delimiter="\t", lineterminator="\r\n"
)
writer.writeheader()
writer.writerows(rows)
summary_content = buffer.getvalue().encode("utf-8")
destination = root / "internal_test_metrics_all_runs.tsv"
summary_status = immutable_publish(destination, summary_content)

audit = {
    "schema_version": "adapted_internal_test_summary_audit_v1",
    "status": "VERIFIED_COMPLETE",
    "arm": "adapted",
    "n_targets": len(expected),
    "internal_test_subjects_per_target": 140,
    "cohort_manifest": {
        "path": project_path(manifest_path),
        "sha256": manifest_sha256,
    },
    "launcher": {
        "path": project_path(launcher_path),
        "sha256": sha256(launcher_path),
    },
    "summary_artifact": {
        "path": project_path(destination),
        "sha256": hashlib.sha256(summary_content).hexdigest(),
        "rows": len(rows),
    },
    "expected_targets": sorted(expected),
    "sources": sources,
}
audit_content = (
    json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n"
).encode("utf-8")
audit_path = root / "internal_test_metrics_all_runs.audit.json"
audit_status = immutable_publish(audit_path, audit_content)
print(f"Summary: {destination} ({summary_status})")
print(f"Audit  : {audit_path} ({audit_status})")
PY

echo "[$(date --iso-8601=seconds)] COMPLETE: 25 adapted checkpoints evaluated once"
