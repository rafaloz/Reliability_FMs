#!/bin/bash
# ============================================================================
# Fix AnatCL preprocessing: re-copy original mwp1*.nii (native CAT12 shape)
# to *_anatcl_ready.nii.gz, replacing the incorrectly resampled 121x128x121.
#
# The original verify_shape() forced everything to 121x128x121 (a training
# crop, not the correct test shape). The model uses AdaptiveAvgPool3d(1)
# so it handles ANY spatial size. We just need the native CAT12 output.
#
# Usage: bash fix_anatcl_shape.sh
# ============================================================================

BASE_DIR="/datos/work/rnavgon/ON-Harmony_Preproc"
PREPROC_DIR="${BASE_DIR}/preprocessings/preproc_AnatCL"

FIXED=0
FAILED=0
SKIPPED=0

echo "============================================================"
echo "  Fixing AnatCL: re-copying original mwp1*.nii (native shape)"
echo "  Replacing incorrectly resampled 121x128x121 files"
echo "============================================================"

for SUBJ_DIR in "${PREPROC_DIR}"/*/; do
    STEM=$(basename "${SUBJ_DIR}")
    READY_FILE="${SUBJ_DIR}${STEM}_anatcl_ready.nii.gz"

    # Find original mwp1*.nii in cat12_work/mri/
    MWP1=$(find "${SUBJ_DIR}cat12_work/mri/" -name "mwp1*.nii" 2>/dev/null | head -1)

    if [ -z "${MWP1}" ]; then
        echo "  SKIP (no mwp1): ${STEM}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if [ ! -f "${READY_FILE}" ]; then
        echo "  SKIP (no ready file): ${STEM}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Re-copy original mwp1 (whatever native CAT12 shape) over the resampled version
    python3 -c "
import nibabel as nib
img = nib.load('${MWP1}')
shape = img.shape[:3]
nib.save(img, '${READY_FILE}')
print(f'  FIXED {shape}: ${STEM}')
" 2>&1

    if [ $? -eq 0 ]; then
        FIXED=$((FIXED + 1))
    else
        echo "  FAIL: ${STEM}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
echo "  Done! Fixed: ${FIXED}, Failed: ${FAILED}, Skipped: ${SKIPPED}"
echo "============================================================"
echo ""
echo "  Verify a sample shape:"
SAMPLE=$(find "${PREPROC_DIR}" -name "*_anatcl_ready.nii.gz" | head -1)
if [ -n "${SAMPLE}" ]; then
    python3 -c "import nibabel as nib; img=nib.load('${SAMPLE}'); print(f'  {img.shape[:3]}: {SAMPLE}')"
fi
