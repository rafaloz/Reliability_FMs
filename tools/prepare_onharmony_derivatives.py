"""Export audited ON-Harmony representations and predictions, without outcomes."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Analyses.figure2_reliability import (
    INPUT_SPECS, METADATA_COLUMNS, canonical_keys, paths_for_spec,
    parse_bids_metadata, prepare_input, select_primary_acquisition_per_cell, sha256_file, validate_manifest,
)
from Analyses.figure3_brain_age import (
    MODEL_SPECS as FROZEN_MODELS, load_onharmony_predictions, validate_prediction_audit,
)
from Analyses.figure4_adaptation import MODEL_SPECS as ADAPTED_MODELS, validate_adapted_onharmony


def export(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".onharmony_export_", dir=args.output.parent))
    inventory = []
    reference = None

    def save(frame, relative, metadata, source_hash, audit_hash):
        nonlocal reference
        frame = frame.copy()
        frame["session_id"] = "ses-" + frame.session_id.astype(str).str.removeprefix("ses-")
        canonical_scanner = parse_bids_metadata(frame.subject_id + "_" + frame.session_id)["scanner"]
        mapping = pd.DataFrame({"original": frame.scanner, "canonical": canonical_scanner}).drop_duplicates()
        if mapping.original.duplicated().any() or mapping.canonical.duplicated().any():
            raise ValueError(f"Scanner mapping is not one-to-one: {relative}")
        frame["scanner"] = canonical_scanner
        keys = canonical_keys(frame)
        if len(frame) != 165 or len(keys.drop_duplicates()) != 165 or frame.subject_id.nunique() != 20:
            raise ValueError(f"Invalid acquisition identities: {relative}")
        if reference is None:
            reference = keys
        elif not keys.equals(reference):
            raise ValueError(f"Acquisition mismatch: {relative}")
        primary = select_primary_acquisition_per_cell(frame)
        if len(primary) != 120 or not primary.groupby("subject_id").size().eq(6).all():
            raise ValueError(f"Invalid primary sample: {relative}")
        primary_keys = set(map(tuple, primary[list(METADATA_COLUMNS)].to_numpy()))
        features = [c for c in frame if c.startswith("emb_")]
        values = features or ["age_pred"]
        if not np.isfinite(frame[values].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite values: {relative}")
        frame = frame[list(METADATA_COLUMNS) + values].copy()
        frame["primary_acquisition"] = [tuple(row) in primary_keys for row in frame[list(METADATA_COLUMNS)].to_numpy()]
        for key, value in metadata.items():
            frame[key] = value
        frame = frame.rename(columns={"subject_id": "participant_id"}).sort_values(["participant_id", "session_id", "scanner"])
        destination = temporary / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
        check = pd.read_csv(destination, float_precision="round_trip")
        np.testing.assert_array_equal(frame[values].to_numpy(), check[values].to_numpy())
        inventory.append(dict(file=str(relative), rows=len(frame), **metadata,
                              source_sha256=source_hash, audit_sha256=audit_hash,
                              csv_sha256=sha256_file(destination)))
        print(f"[{len(inventory)}] {relative}", flush=True)

    try:
        for spec in INPUT_SPECS:
            if spec.condition == "freesurfer_combat":
                continue
            paths, audits = paths_for_spec(spec, args.project / "DATA/embeddings",
                                          args.project / "Results/freesurfer_thalamus_merge_201/lme4_inputs",
                                          args.project / "DATA/embeddings_bfloat16_v2")
            for fold, (source, audit) in enumerate(zip(paths, audits)):
                provenance = validate_manifest(audit, source, spec=spec, fold_index=fold)
                scratch = temporary / "_input.csv"
                frame, _ = prepare_input(source, spec=spec, destination=scratch)
                scratch.unlink()
                if sha256_file(source) != provenance["data_sha256"] or sha256_file(audit) != provenance["sha256"]:
                    raise ValueError(f"Source changed: {source}")
                variant = f"fold{fold}" if len(paths) > 1 else "canonical"
                save(frame, Path("embeddings") / spec.condition / variant / "onharmony.csv",
                     dict(model=spec.label, variant=variant, arm="embedding", run=variant, primary_run=fold in spec.primary_folds),
                     provenance["data_sha256"], provenance["sha256"])
        brain_age = args.project / "Results/brain_age"
        for spec in FROZEN_MODELS:
            variants = tuple(f"AnatCL/fold{i}" for i in range(5)) if spec.condition == "anatcl" else spec.variants
            for variant_path in variants:
                model, variant = variant_path.split("/")
                root = brain_age / "onharmony" / ("frozen_bfloat16_v3" if model == "AnatCL" else "frozen") / variant_path
                path, audit = root / "predictions.tsv", root / "prediction_audit.json"
                source_hash, audit_hash = sha256_file(path), sha256_file(audit)
                validate_prediction_audit(audit, path, spec=spec, variant=variant_path)
                frame = load_onharmony_predictions(path, spec=spec, variant=variant_path)
                frame["session_id"] = frame.session
                if sha256_file(path) != source_hash or sha256_file(audit) != audit_hash:
                    raise ValueError(f"Source changed: {path}")
                save(frame, Path("predictions") / model / f"ridge_{variant}" / "onharmony.csv",
                     dict(model=spec.label, variant=variant, arm="frozen", run=f"ridge_{variant}", primary_run=variant_path in spec.variants), source_hash, audit_hash)
        for spec in ADAPTED_MODELS:
            for run in spec.runs:
                root = brain_age / "onharmony/adapted" / spec.model_dir / run.run
                path, audit = root / "predictions.tsv", root / "prediction_audit.json"
                source_hash, audit_hash = sha256_file(path), sha256_file(audit)
                payload = json.loads(audit.read_text())
                frame, _ = validate_adapted_onharmony(audit, path, spec=spec, run=run,
                    checkpoint_sha256=payload["adapted_checkpoint_sha256"])
                frame["session_id"] = frame.session
                if sha256_file(path) != source_hash or sha256_file(audit) != audit_hash:
                    raise ValueError(f"Source changed: {path}")
                save(frame, Path("predictions") / spec.model_dir / run.run / "onharmony.csv",
                     dict(model=spec.label, variant=run.adapted_variant, arm="adapted", run=run.run,
                          seed=run.seed, fold=run.fold if run.fold is not None else "", primary_run=run.primary),
                     source_hash, audit_hash)
        if len(inventory) != 53:
            raise ValueError(f"Expected 53 files, found {len(inventory)}")
        pd.DataFrame(inventory).to_csv(temporary / "inventory.csv", index=False)
        definitions = {
            "participant_id": "Original public ON-Harmony study identifier.",
            "session_id": "Original BIDS session identifier with ses- prefix.",
            "scanner": "Scanner code used in the study analyses.",
            "primary_acquisition": "True for the 120 first acquisitions used in primary analyses; all 165 are included.",
            "primary_run": "AnatCL and AnatCL-SimCLR use fold0 for primary analyses; AnatCL fine-tuning uses all three fold0 seeds. Other folds are sensitivity runs.",
            "emb_*": "Representation coordinates, not aligned across models or folds; FreeSurfer uses 201 morphometric coordinates.",
            "age_pred": "Predicted age in years, not clipped; no true age or prediction errors are included.",
            "model/variant/arm/run/seed/fold": "Model and retained-run identity; embeddings and predictions are never averaged across runs.",
            "source_sha256/audit_sha256/csv_sha256": "Content digests of source, checked audit and exported CSV, respectively.",
        }
        pd.DataFrame(definitions.items(), columns=["column", "description"]).to_csv(temporary / "data_dictionary.csv", index=False)
        temporary.rename(args.output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(f"PASS: ON-Harmony deposit in {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    export(parser.parse_args())
