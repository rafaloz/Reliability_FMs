"""Prepare a CSV-only derivative deposit for one development cohort; never upload."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Analyses.figure3_brain_age import MODEL_SPECS as FROZEN_MODELS
from Analyses.figure4_adaptation import MODEL_SPECS as ADAPTED_MODELS

COUNTS = {"train": 521, "validation": 65, "internal_test": 64}
COHORT_COUNTS = {
    "CamCAN": COUNTS,
    "MR-ART": {"train": 118, "validation": 15, "internal_test": 15},
    "NIMH-RV": {"train": 118, "validation": 14, "internal_test": 14},
    "NeuroCog": {"train": 239, "validation": 30, "internal_test": 30},
    "WAND": {"train": 134, "validation": 17, "internal_test": 17},
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_frame(path, expected, split, *, embeddings=False, dataset="CamCAN"):
    before = sha(path)
    columns = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    features = [c for c in columns if c.startswith("emb_")] if embeddings else ["age_pred"]
    if embeddings and (len(features) not in (512, 768, 1024) or features != [f"emb_{i}" for i in range(len(features))]):
        raise ValueError(f"Invalid embedding columns: {path}")
    frame = pd.read_csv(path, sep="\t", usecols=["dataset", "subject", "split"] + features)
    if not frame["split"].eq(split).all():
        raise ValueError(f"Wrong split: {path}")
    frame = frame.loc[frame.dataset.eq(dataset), ["subject", "split"] + features]
    if frame.subject.duplicated().any() or set(frame.subject) != expected:
        raise ValueError(f"{dataset} subject mismatch: {path}")
    if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite derivatives: {path}")
    if sha(path) != before:
        raise ValueError(f"Source changed during read: {path}")
    return frame.sort_values("subject").reset_index(drop=True).rename(columns={"subject": "participant_id"}), before


def export(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = pd.read_csv(args.manifest, sep="\t", usecols=["dataset", "subject", "split"])
    dataset = getattr(args, "dataset", "CamCAN")
    counts = COHORT_COUNTS[dataset]
    cohort = manifest.loc[manifest.dataset.eq(dataset)]
    expected = {split: set(cohort.loc[cohort.split.eq(split), "subject"]) for split in counts}
    if cohort.subject.duplicated().any() or len(cohort) != sum(counts.values()) or any(len(expected[s]) != n for s, n in counts.items()):
        raise ValueError(f"Unexpected {dataset} participant counts: expected {counts}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".derivative_deposit_", dir=args.output.parent))
    inventory = []

    def save(path, split, model, variant, arm, run, seed="", fold="", primary=True, audit=None):
        frame, source_sha = select_frame(path, expected[split], split, embeddings=arm == "embedding", dataset=dataset)
        audit_sha = ""
        readout, amp = "", ""
        if audit is not None:
            audit_sha = sha(audit)
            record = json.loads(audit.read_text())
            key = "artifact_sha256" if arm == "embedding" else "predictions_artifact_sha256"
            expected_sha = record.get(key)
            if arm != "embedding" and expected_sha is None:
                metrics_path = path.parent / "metrics_internal_test.json"
                if sha(metrics_path) != record.get("result_sha256"):
                    raise ValueError(f"Receipt/metrics mismatch: {metrics_path}")
                expected_sha = json.loads(metrics_path.read_text()).get(key)
            if expected_sha != source_sha:
                raise ValueError(f"Source/audit mismatch: {path} ({key})")
            if arm == "embedding" and record.get("embedding_dim") != sum(c.startswith("emb_") for c in frame):
                raise ValueError(f"Embedding dimension/audit mismatch: {path}")
            if record.get("variant") != variant:
                raise ValueError(f"Variant/audit mismatch: {path}")
            if sha(audit) != audit_sha:
                raise ValueError(f"Audit changed: {audit}")
            readout = record.get("readout", "")
            amp = record.get("runtime", {}).get("amp", "")
        metadata = dict(model=model, variant=variant, arm=arm, run=run, seed=seed, fold=fold, primary=primary)
        for key, value in metadata.items():
            frame[key] = value
        features = [c for c in frame if c.startswith("emb_")] if arm == "embedding" else ["age_pred"]
        frame = frame[["participant_id", "split"] + list(metadata) + features]
        relative = Path("embeddings" if arm == "embedding" else "predictions") / model / run / f"{split}.csv"
        destination = temporary / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
        reloaded = pd.read_csv(destination, float_precision="round_trip")
        np.testing.assert_array_equal(frame[features].to_numpy(), reloaded[features].to_numpy())
        if reloaded.participant_id.tolist() != frame.participant_id.tolist():
            raise ValueError("CSV identities changed")
        inventory.append(dict(file=relative.as_posix(), rows=len(frame), **metadata, split=split,
                              readout=readout, amp=amp, source_sha256=source_sha,
                              audit_sha256=audit_sha, csv_sha256=sha(destination)))
        print(f"[{len(inventory)}] {relative}: {len(frame)} rows", flush=True)

    try:
        for spec in FROZEN_MODELS:
            variants = tuple(f"AnatCL/fold{i}" for i in range(5)) if spec.condition == "anatcl" else spec.variants
            for variant_path in variants:
                model, variant = variant_path.split("/")
                fold = variant.removeprefix("fold") if model == "AnatCL" else ""
                suffix = "_bfloat16_v3" if model == "AnatCL" else ""
                embedding_path = "BrainSegFounder/canonical" if model == "BrainSegFounder" else variant_path
                for split in counts:
                    path = args.brain_age_root / f"embeddings_FT{suffix}" / embedding_path / f"{split}.tsv"
                    save(path, split, model, variant, "embedding", variant, fold=fold, primary=variant_path in spec.variants,
                         audit=Path(str(path) + ".audit.json"))
                fit = args.brain_age_root / f"frozen{suffix}" / variant_path
                for split in ("validation", "internal_test"):
                    path = fit / "locked_validation_predictions.tsv" if split == "validation" else (
                        args.brain_age_root / "internal_test" / f"frozen{suffix}" / variant_path / "predictions_internal_test.tsv")
                    save(path, split, model, variant, "frozen", f"ridge_{variant}", fold=fold, primary=variant_path in spec.variants,
                         audit=fit / "INTERNAL_TEST_EVALUATED.json" if split == "internal_test" else None)
        for spec in ADAPTED_MODELS:
            for run in spec.runs:
                fit = args.brain_age_root / "adapted" / spec.model_dir / run.run
                for split in ("validation", "internal_test"):
                    path = fit / "locked_validation_predictions.tsv" if split == "validation" else (
                        args.brain_age_root / "internal_test/adapted" / spec.model_dir / run.run / "predictions_internal_test.tsv")
                    save(path, split, spec.model_dir, run.adapted_variant, "adapted", run.run,
                         seed=run.seed, fold=run.fold if run.fold is not None else "", primary=run.primary,
                         audit=fit / "INTERNAL_TEST_EVALUATED.json" if split == "internal_test" else None)
        pd.DataFrame(inventory).to_csv(temporary / "inventory.csv", index=False)
        definitions = {
            "participant_id": f"Original {dataset} study identifier; not a newly anonymised identifier.",
            "split": "Original development split: train, validation or internal_test.",
            "model": "Foundation-model identifier.", "variant": "Encoder variant or AnatCL pretrained fold.",
            "arm": "embedding, frozen Ridge prediction, or adapted prediction.",
            "run": "Representation or retained training-run identifier.",
            "seed": "Training seed; blank for frozen embeddings and Ridge.",
            "fold": "AnatCL pretrained fold 0-4; otherwise blank.",
            "primary": "AnatCL uses fold0 in primary analyses, with all three fine-tuning seeds; folds1-4 are sensitivity runs. No predictions are ensembled.",
            "emb_*": "Frozen encoder feature values; axes are not interchangeable across models or folds.",
            "age_pred": "Predicted age in years, inverse-transformed with training statistics; not clipped.",
            "file": "Relative path within this deposit.", "rows": "Number of cohort rows in the CSV.",
            "readout": "Embedding readout recorded by source audit; blank when not exported.",
            "amp": "Source embedding forward precision; off means FP32; blank means not exported.",
            "source_sha256": "Digest of complete source TSV, not redistributed here.",
            "audit_sha256": "Digest of source audit when checked; validation predictions have no audit digest here.",
            "csv_sha256": "Digest of exported CSV.",
        }
        pd.DataFrame(definitions.items(), columns=["column", "description"]).to_csv(temporary / "data_dictionary.csv", index=False)
        notes = {
            "status": "CSV derivative deposit for the accompanying study.",
            "scope": f"{dataset} derivatives; seven models, eleven frozen representations and 25 retained adapted runs.",
            "subjects": f"{sum(counts.values())} unique participants: {counts}.",
            "excluded": "No raw MRI, true ages, sex, error/BAG columns, source demographics or filesystem paths.",
            "limitations": "Not a complete reproduction dataset; excludes other cohorts, processed images and model weights.",
            "source_validation": "Embeddings checked against sidecars; test predictions against receipts; validation predictions checked for identity, split and finiteness, with source digests recorded.",
            "identifier_policy": "Original study IDs retained for linkage; these are not newly anonymised identifiers.",
            "manifest_sha256": sha(args.manifest), "export_script_sha256": sha(Path(__file__)),
        }
        pd.DataFrame(notes.items(), columns=["item", "description"]).to_csv(temporary / "deposit_notes.csv", index=False)
        if len(inventory) != 105:
            raise ValueError(f"Expected 105 data CSVs, found {len(inventory)}")
        temporary.rename(args.output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(f"PASS: 105 data CSVs + 3 documentation CSVs in {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-age-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=COHORT_COUNTS, default="CamCAN")
    export(parser.parse_args())
