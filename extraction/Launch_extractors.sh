#!/bin/bash
# ============================================================================
# Launch all 5 embedding extractors on preprocessed ON-Harmony data
# ============================================================================
# Usage:  bash Launch_extractors.sh
#
# Expects:
#   - Preprocessed files in PREPROC_DIR/preproc_{model}/
#   - Model weights in MODELS_DIR/
#   - Extraction scripts in same directory as this script
# ============================================================================

set -e

# ── Configurable paths ──────────────────────────────────────────────────────
BASE_DIR="/datos/work/rnavgon/ON-Harmony_Preproc"
PREPROC_DIR="${BASE_DIR}/preprocessings"
MODELS_DIR="${BASE_DIR}/Models"
OUTPUT_DIR="${BASE_DIR}/DATA"
SCRIPTS_DIR="${BASE_DIR}/Emb_extractors"
PARTICIPANTS="${OUTPUT_DIR}/participants.tsv"  # in DATA/ folder

# Device: auto-detect CUDA
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
    DEVICE="cuda"
else
    DEVICE="cpu"
fi
echo "Device: ${DEVICE}"

mkdir -p "${OUTPUT_DIR}"

# ── Helper: build file list for a model ─────────────────────────────────────
build_file_list() {
    local model_tag="$1"    # e.g. BrainIAC
    local preproc_subdir="$2"  # e.g. preproc_BrainIAC
    local suffix="$3"       # e.g. _brainiac_ready.nii.gz
    local list_file="${OUTPUT_DIR}/filelist_${model_tag}.txt"

    > "${list_file}"
    for d in "${PREPROC_DIR}/${preproc_subdir}"/sub-*; do
        [ -d "$d" ] || continue
        stem=$(basename "$d")
        target="${d}/${stem}${suffix}"
        if [ -f "$target" ]; then
            echo "$target" >> "${list_file}"
        else
            echo "  WARNING: missing ${target}" >&2
        fi
    done

    local count=$(wc -l < "${list_file}")
    echo "  File list: ${list_file} (${count} files)" >&2
    echo "${list_file}"
}

# ── Participants flag ────────────────────────────────────────────────────────
PART_FLAG=""
if [ -n "${PARTICIPANTS}" ] && [ -f "${PARTICIPANTS}" ]; then
    PART_FLAG="--participants ${PARTICIPANTS}"
    echo "Participants file: ${PARTICIPANTS}"
else
    echo "No participants.tsv found — demographics will be empty"
fi

echo ""
echo "============================================================"
echo "  Embedding extraction — ON-Harmony"
echo "============================================================"
echo ""

# ── 1. BrainIAC (ViT-B, 768-dim) ──────────────────────────────────────────
echo "=== BrainIAC ==="
CKPT="${MODELS_DIR}/BrainIAC/BrainIAC.ckpt"
if [ -f "$CKPT" ]; then
    LIST=$(build_file_list BrainIAC preproc_BrainIAC _brainiac_ready.nii.gz)
    python3 "${SCRIPTS_DIR}/extract_BrainIAC.py" \
        --input_list "${LIST}" \
        --checkpoint "${CKPT}" \
        ${PART_FLAG} \
        --output "${OUTPUT_DIR}/embeddings_BrainIAC.csv" \
        --device "${DEVICE}"
    echo ""
else
    echo "  SKIP: checkpoint not found at ${CKPT}"
    echo ""
fi

# ── 2. 3D-Neuro-SimCLR (ResNet-18, 512-dim) ───────────────────────────────
echo "=== 3D-Neuro-SimCLR ==="
CKPT="${MODELS_DIR}/3D-Neuro-SimCLR/simclr_3d_brain_foundation.tar"
if [ -f "$CKPT" ]; then
    LIST=$(build_file_list 3DNeuroSimCLR preproc_SimCLR _simclr_ready.nii.gz)
    python3 "${SCRIPTS_DIR}/extract_3DNeuroSimCLR.py" \
        --input_list "${LIST}" \
        --checkpoint "${CKPT}" \
        ${PART_FLAG} \
        --output "${OUTPUT_DIR}/embeddings_3DNeuroSimCLR.csv" \
        --device "${DEVICE}"
    echo ""
else
    echo "  SKIP: checkpoint not found at ${CKPT}"
    echo ""
fi

# ── 3. AnatCL (ResNet-18, 512-dim, 5 folds) ──────────────────────────────
echo "=== AnatCL ==="
ANATCL_ZIP="${MODELS_DIR}/AnatCL/anatcl-pretrained-main-resnet18-anatcl-g3.zip"
ANATCL_FOLD_DIR="${MODELS_DIR}/AnatCL/resnet18-anatcl-g3"
# Extract zip if fold .pth files not already present
if [ ! -f "${ANATCL_FOLD_DIR}/fold0.pth" ] && [ -f "${ANATCL_ZIP}" ]; then
    echo "  Extracting AnatCL weights from zip..."
    unzip -o -q "${ANATCL_ZIP}" -d "${MODELS_DIR}/AnatCL/"
    # Flatten: zip extracts into anatcl-pretrained-main-resnet18-anatcl-g3/resnet18-anatcl-g3/
    if [ -d "${MODELS_DIR}/AnatCL/anatcl-pretrained-main-resnet18-anatcl-g3/resnet18-anatcl-g3" ]; then
        mv "${MODELS_DIR}/AnatCL/anatcl-pretrained-main-resnet18-anatcl-g3/resnet18-anatcl-g3" \
           "${ANATCL_FOLD_DIR}"
        rmdir "${MODELS_DIR}/AnatCL/anatcl-pretrained-main-resnet18-anatcl-g3" 2>/dev/null || true
    fi
fi
if [ -f "${ANATCL_FOLD_DIR}/fold0.pth" ]; then
    LIST=$(build_file_list AnatCL preproc_AnatCL _anatcl_ready.nii.gz)
    if [ -s "${LIST}" ]; then
        python3 "${SCRIPTS_DIR}/extract_AnatCL.py" \
            --input_list "${LIST}" \
            --checkpoint_dir "${ANATCL_FOLD_DIR}" \
            --descriptor global \
            ${PART_FLAG} \
            --output_dir "${OUTPUT_DIR}" \
            --device "${DEVICE}"
    else
        echo "  SKIP: no preprocessed files found"
    fi
    echo ""
else
    echo "  SKIP: AnatCL weights not found (no zip or fold0.pth in ${ANATCL_FOLD_DIR})"
    echo ""
fi

# ── 4. y-Aware (DenseNet-121, 1024-dim) ───────────────────────────────────
echo "=== y-Aware ==="
CKPT="${MODELS_DIR}/y-Aware/DenseNet121_BHB-10K_yAwareContrastive.pth"
if [ -f "$CKPT" ]; then
    LIST=$(build_file_list yAware preproc_yAware _yaware_ready.npy)
    python3 "${SCRIPTS_DIR}/extract_yAware.py" \
        --input_list "${LIST}" \
        --checkpoint "${CKPT}" \
        ${PART_FLAG} \
        --output "${OUTPUT_DIR}/embeddings_yAware.csv" \
        --device "${DEVICE}"
    echo ""
else
    echo "  SKIP: checkpoint not found at ${CKPT}"
    echo ""
fi

# ── 5. BrainSegFounder (Swin-T, 768-dim) ──────────────────────────────────
echo "=== BrainSegFounder ==="
CKPT="${MODELS_DIR}/BrainSegFounder/64-gpu-model_bestValRMSE.pt"
if [ -f "$CKPT" ]; then
    LIST=$(build_file_list BrainSegFounder preproc_BrainSegFounder _brainsegfounder_ready.nii.gz)
    python3 "${SCRIPTS_DIR}/extract_BrainSegFounder.py" \
        --input_list "${LIST}" \
        --checkpoint "${CKPT}" \
        ${PART_FLAG} \
        --output "${OUTPUT_DIR}/embeddings_BrainSegFounder.csv" \
        --device "${DEVICE}"
    echo ""
else
    echo "  SKIP: checkpoint not found at ${CKPT}"
    echo ""
fi

echo "============================================================"
echo "  All done! Output CSVs in: ${OUTPUT_DIR}/"
echo "============================================================"
ls -lh "${OUTPUT_DIR}"/embeddings_*.csv 2>/dev/null || echo "  (no CSVs produced)"
