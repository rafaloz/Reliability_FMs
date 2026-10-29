#!/usr/bin/env python3
"""Recalculate study metrics directly from the compact published CSVs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Analyses import figure2_reliability as embedding
from Analyses import figure3_brain_age as frozen
from Analyses import figure4_adaptation as adaptation
from FineTuning.shared.metrics import regression_metrics
from tools.prepare_reproduction_selections import COUNTS, digest, require, verify


SCANNERS = {
    "NOT1ACH": "Ach", "NOT2ING": "Ing", "NOT3GEM": "750",
    "NOT4GEP": "Prem48", "OXF1PRI": "Pri32", "OXF2PRI": "Pri64",
    "OXF3TRI": "Trio", "OXF4GEP": "Prem21",
}
ON_KEYS = ["participant_id", "session_id", "scanner"]
TEST_KEYS = ["dataset", "subject"]
SEEDS = {"onharmony": 42, "internal_test": 31415, "scanner_effects": 2718}
GAIN_METRICS = {
    "accuracy_gain": ("mae", False),
    "reliability_gain": ("dbicc", True),
    "stability_gain": ("scanner_difference", False),
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, float_precision="round_trip")


def checked_ages(path: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["dataset", "subject", "age"],
                        dtype={"dataset": str, "subject": str}, float_precision="round_trip")
    require(not frame[TEST_KEYS].isna().any().any() and not frame.duplicated(TEST_KEYS).any(),
            "Age CSV contains missing or duplicate subject identifiers")
    frame["age"] = pd.to_numeric(frame["age"], errors="raise")
    require(np.isfinite(frame["age"].to_numpy(dtype=float)).all(), "Age CSV contains nonfinite ages")
    require(frame["age"].between(18, 88).all(), "Development ages must be in the study range 18-88")
    allowed = pd.MultiIndex.from_frame(cohort[TEST_KEYS])
    require(pd.MultiIndex.from_frame(frame[TEST_KEYS]).isin(allowed).all(),
            "Age CSV contains subjects outside the published development cohort")
    test = cohort.loc[cohort["split"].eq("internal_test"), TEST_KEYS]
    result = test.merge(frame, on=TEST_KEYS, how="left", validate="one_to_one")
    require(len(result) == 140 and result["age"].notna().all(),
            "Age CSV must contain all 140 internal-test subjects")
    return result.sort_values(TEST_KEYS, kind="stable").reset_index(drop=True)


def select_run(frame: pd.DataFrame, *, model: str, variant: str, arm: str,
               run: str, primary: bool, seed: int | None = None, fold: int | None = None) -> pd.DataFrame:
    selected = frame.loc[frame["model"].eq(model) & frame["variant"].eq(variant)
                         & frame["arm"].eq(arm) & frame["run"].eq(run)].copy()
    require(len(selected) > 0, f"Missing published run: {model}/{variant}/{arm}/{run}")
    primary_column = "primary_run" if "primary_run" in selected else "primary"
    require(selected[primary_column].astype(str).eq(str(primary)).all(),
            f"Wrong primary-run flag for {model}/{run}")
    for column, value in (("seed", seed), ("fold", fold)):
        if value is not None:
            require(pd.to_numeric(selected[column], errors="raise").eq(value).all(),
                    f"Wrong {column} for {model}/{run}")
    require(np.isfinite(selected["age_pred"].to_numpy(dtype=float)).all(),
            f"Nonfinite predictions for {model}/{run}")
    return selected


def onharmony_frame(frame: pd.DataFrame, acquisitions: pd.DataFrame) -> pd.DataFrame:
    require(len(frame) == 165 and not frame.duplicated(ON_KEYS[:2]).any(),
            "A prediction run must contain 165 distinct acquisitions")
    expected = acquisitions[ON_KEYS + ["primary_acquisition", "phase"]]
    compared = frame.merge(expected, on=ON_KEYS, how="outer", suffixes=("", "_expected"),
                           validate="one_to_one", indicator=True)
    require(compared["_merge"].eq("both").all(), "ON-Harmony acquisition identities differ")
    require(compared["primary_acquisition"].astype(str).eq(
        compared["primary_acquisition_expected"].astype(str)).all(), "Wrong primary-acquisition flag")
    result = compared.rename(columns={"participant_id": "subject_id", "phase": "cohort"})
    result["scanner"] = result["scanner"].map(SCANNERS)
    require(result["scanner"].notna().all(), "Unknown ON-Harmony scanner code")
    return result.sort_values(["subject_id", "session_id", "scanner"], kind="stable").reset_index(drop=True)


def internal_metrics(frame: pd.DataFrame, ages: pd.DataFrame, samples: np.ndarray) -> dict:
    predicted = frame.rename(columns={"participant_id": "subject"})
    require(len(predicted) == 140 and not predicted.duplicated(TEST_KEYS).any(),
            "A run must contain 140 distinct internal-test subjects")
    joined = ages.merge(predicted[TEST_KEYS + ["age_pred"]], on=TEST_KEYS, how="outer",
                        validate="one_to_one", indicator=True).sort_values(TEST_KEYS, kind="stable")
    require(joined["_merge"].eq("both").all(), "Internal-test subjects differ from the age table")
    truth, prediction = joined["age"].to_numpy(float), joined["age_pred"].to_numpy(float)
    mae = adaptation.scalar_metric(np.abs(prediction - truth), samples, name="internal-test MAE")
    point_r2 = regression_metrics(truth, prediction)["r2"]
    r2_bootstrap = np.asarray([regression_metrics(truth[index], prediction[index])["r2"]
                               for index in samples])
    return {"mae": mae, "r2": adaptation.metric_record(point_r2, r2_bootstrap, name="internal-test R2")}


def prediction_metrics(frame: pd.DataFrame, samples: np.ndarray) -> tuple[dict, pd.DataFrame]:
    cells = frozen.participant_scanner_cells(frame)
    difference = frozen.pairwise_absolute_difference_by_subject(cells)
    require(difference.index.tolist() == sorted(frame["subject_id"].unique()), "Bootstrap subject order differs")
    return {
        "dbicc": embedding.compute_dbicc(cells, ["age_pred"], samples),
        "scanner_difference": adaptation.scalar_metric(difference.to_numpy(), samples, name="scanner difference"),
    }, cells


def aggregate_metrics(records: list[dict], name: str) -> dict:
    return {metric: adaptation.aggregate_metric_records([record[metric] for record in records],
                                                        name=f"{name} {metric}")
            for metric in records[0]}


def absolute_values(metrics: dict) -> dict:
    return {"mae_years": metrics["mae"]["estimate"] if "mae" in metrics else None,
            "prediction_dbicc": metrics["dbicc"]["estimate"],
            "scanner_difference_years": metrics["scanner_difference"]["estimate"],
            "r2": metrics["r2"]["estimate"] if "r2" in metrics else None}


def representation_analysis(data_root: Path, output: Path, samples: np.ndarray, args) -> dict:
    conditions = {}
    for spec in embedding.INPUT_SPECS:
        if spec.condition == "freesurfer_combat":
            continue
        path = data_root / "ON-Harmony/embeddings" / f"{spec.condition}.csv"
        frame = read_csv(path)
        features = embedding.numbered_columns(frame, "emb_")
        require(len(features) == spec.expected_dimensions, f"Wrong embedding dimension: {path}")
        variants = [f"fold{i}" for i in range(5)] if spec.condition in {"anatcl", "matched_simclr"} else ["canonical"]
        require(set(frame["variant"]) == set(variants), f"Wrong embedding variants: {path}")
        fold_dbicc, fold_variance = [], []
        for index, variant in enumerate(variants):
            group = frame.loc[frame["variant"].eq(variant)].rename(columns={"participant_id": "subject_id"})
            require(len(group) == 165, f"Wrong embedding row count: {path}/{variant}")
            group = group[list(embedding.METADATA_COLUMNS) + features].sort_values(
                list(embedding.METADATA_COLUMNS), kind="stable").reset_index(drop=True)
            fold_dbicc.append(embedding.compute_dbicc(group, features, samples))
            if args.reml:
                input_path = output / "inputs" / f"{spec.condition}__fold{index}.csv"
                embedding.atomic_csv(input_path, group)
                result_path = output / "reml" / f"{spec.condition}__fold{index}.json"
                result_path.parent.mkdir(parents=True, exist_ok=True)
                fold_variance.append(embedding.run_reml(input_path, result_path,
                    condition=f"{spec.condition}__fold{index}", workers=args.workers, rscript=args.rscript))
        conditions[spec.condition] = {
            "label": spec.label, "role": spec.role,
            "primary_folds": list(spec.primary_folds),
            "dbicc": embedding.aggregate_fold_results([fold_dbicc[i] for i in spec.primary_folds]), "fold_dbicc": fold_dbicc,
            "variance": embedding.aggregate_trace_results([fold_variance[i] for i in spec.primary_folds]) if fold_variance else None,
            "fold_variance": fold_variance,
            "fold_sensitivity": {
                "dbicc": embedding.aggregate_fold_results(fold_dbicc),
                "variance": embedding.aggregate_trace_results(fold_variance) if fold_variance else None,
            } if len(fold_dbicc) > 1 else None,
        }
        print(f"Embedding: {spec.label}", flush=True)
    return {"schema_version": 1, "conditions": conditions, "contrasts": {
        "anatcl_minus_matched_simclr": embedding.paired_bootstrap_contrast(
            conditions["anatcl"]["dbicc"], conditions["matched_simclr"]["dbicc"], label="AnatCL fold0 minus AnatCL-SimCLR fold0"),
        "anatcl_minus_matched_simclr_all_folds": embedding.paired_bootstrap_contrast(
            conditions["anatcl"]["fold_sensitivity"]["dbicc"], conditions["matched_simclr"]["fold_sensitivity"]["dbicc"],
            label="AnatCL minus AnatCL-SimCLR: five matched folds sensitivity")}}


def prediction_analysis(data_root: Path, acquisitions: pd.DataFrame, cohort: pd.DataFrame,
                        representations: dict, ages: pd.DataFrame | None, samples: dict) -> tuple[dict, dict]:
    onharmony = read_csv(data_root / "ON-Harmony/predictions.csv")
    internal = []
    for dataset in COUNTS:
        frame = read_csv(data_root / dataset / "predictions.csv")
        frame = frame.loc[frame["split"].eq("internal_test")].copy()
        frame["dataset"] = dataset
        internal.append(frame)
    internal = pd.concat(internal, ignore_index=True)
    expected_test = set(map(tuple, cohort.loc[cohort["split"].eq("internal_test"), TEST_KEYS].to_numpy()))
    conditions_frozen, conditions_adapted, scanner_cells = {}, {}, {}
    for spec in adaptation.MODEL_SPECS:
        cache, runs, primary = {}, {}, []
        primary_variants = {run.frozen_variant for run in spec.primary_runs}
        for run in spec.runs:
            variant = run.frozen_variant
            if variant not in cache:
                frozen_on = select_run(onharmony, model=spec.label, variant=variant, arm="frozen",
                                       run=f"ridge_{variant}", primary=variant in primary_variants)
                frozen_test = select_run(internal, model=spec.model_dir, variant=variant, arm="frozen",
                                         run=f"ridge_{variant}", primary=variant in primary_variants)
                require(set(zip(frozen_test.dataset, frozen_test.participant_id)) == expected_test,
                        f"Wrong frozen test identities: {spec.label}/{variant}")
                metrics, cells = prediction_metrics(onharmony_frame(frozen_on, acquisitions), samples["onharmony"])
                if ages is not None:
                    metrics.update(internal_metrics(frozen_test, ages, samples["internal_test"]))
                cache[variant] = {"metrics": metrics, "cells": cells}
            frozen_metrics = cache[variant]["metrics"]
            adapted_on = select_run(onharmony, model=spec.label, variant=run.adapted_variant, arm="adapted",
                                    run=run.run, primary=run.primary, seed=run.seed, fold=run.fold)
            adapted_test = select_run(internal, model=spec.model_dir, variant=run.adapted_variant, arm="adapted",
                                      run=run.run, primary=run.primary, seed=run.seed, fold=run.fold)
            require(set(zip(adapted_test.dataset, adapted_test.participant_id)) == expected_test,
                    f"Wrong adapted test identities: {spec.label}/{run.run}")
            adapted_metrics, adapted_cells = prediction_metrics(onharmony_frame(adapted_on, acquisitions), samples["onharmony"])
            if ages is not None:
                adapted_metrics.update(internal_metrics(adapted_test, ages, samples["internal_test"]))
            gains = {endpoint: adaptation.contrast_metric(frozen_metrics[metric], adapted_metrics[metric],
                     adapted_minus_frozen=direction, name=f"{spec.label}/{run.run}/{endpoint}")
                     for endpoint, (metric, direction) in GAIN_METRICS.items() if metric in frozen_metrics}
            runs[run.run] = {
                "run": run.run, "seed": run.seed, "fold": run.fold, "primary": run.primary,
                "variant": run.adapted_variant, "frozen_variant": variant,
                "absolute": {"frozen": absolute_values(frozen_metrics), "adapted": absolute_values(adapted_metrics)},
                "gains": {key: value["estimate"] for key, value in gains.items()},
            }
            if run.primary:
                primary.append({"frozen": frozen_metrics, "adapted": adapted_metrics,
                                "gains": gains, "cells": adapted_cells})
            print(f"Predictions: {spec.label}/{run.run}", flush=True)
        frozen_aggregate = aggregate_metrics([record["frozen"] for record in primary], f"{spec.label} frozen")
        adapted_aggregate = aggregate_metrics([record["adapted"] for record in primary], f"{spec.label} adapted")
        gains = {endpoint: adaptation.aggregate_contrasts([record["gains"][endpoint] for record in primary], name=endpoint)
                 for endpoint in primary[0]["gains"]}
        conditions_adapted[spec.condition] = {
            "label": spec.label, "primary_definition": spec.primary_definition,
            "primary_runs": [run.run for run in spec.primary_runs], "run_results": runs,
            "absolute": {"frozen": frozen_aggregate, "adapted": adapted_aggregate}, "primary_gains": gains,
        }
        embedding_dbicc = representations["conditions"][spec.condition]["dbicc"]
        change = embedding.paired_bootstrap_contrast(frozen_aggregate["dbicc"], embedding_dbicc,
                                                    label=f"{spec.label}: prediction minus embedding dbICC")
        conditions_frozen[spec.condition] = {
            "label": spec.label, "internal_test_mae": frozen_aggregate.get("mae"),
            "internal_test_r2": frozen_aggregate.get("r2"), "embedding_dbicc": embedding_dbicc,
            "prediction_dbicc": frozen_aggregate["dbicc"], "dbicc_change": change,
            "cross_scanner_absolute_difference": frozen_aggregate["scanner_difference"],
        }
        scanner_cells[f"frozen:{spec.condition}"] = [
            pair["cells"] for variant, pair in cache.items() if variant in primary_variants
        ]
        scanner_cells[f"adapted:{spec.condition}"] = [record["cells"] for record in primary]
    participants = acquisitions[["participant_id", "phase"]].drop_duplicates()
    scanner_samples = frozen.stratified_phase_bootstrap(participants, len(samples["onharmony"]), SEEDS["scanner_effects"])
    scanner_effects = adaptation.compute_scanner_effect_bootstraps_batched(scanner_cells, scanner_samples, frozen.SCANNER_ORDER)
    for condition in conditions_frozen:
        conditions_frozen[condition]["scanner_effects"] = scanner_effects[f"frozen:{condition}"]
        conditions_adapted[condition]["adapted_scanner_effects"] = scanner_effects[f"adapted:{condition}"]
    adjusted = frozen.benjamini_hochberg({key: value["dbicc_change"]["p_value_two_sided"]
                                        for key, value in conditions_frozen.items()})
    for condition, value in adjusted.items():
        conditions_frozen[condition]["dbicc_change"]["q_value_bh"] = value
    all_p = {}
    for endpoint in GAIN_METRICS:
        family = {key: value["primary_gains"][endpoint]["p_value_two_sided"]
                  for key, value in conditions_adapted.items() if endpoint in value["primary_gains"]}
        for condition, q in frozen.benjamini_hochberg(family).items():
            conditions_adapted[condition]["primary_gains"][endpoint]["q_value_bh_within_endpoint"] = q
            all_p[f"{endpoint}:{condition}"] = family[condition]
    global_q = frozen.benjamini_hochberg(all_p) if len(all_p) == 21 else {}
    for condition, value in conditions_adapted.items():
        for endpoint, gain in value["primary_gains"].items():
            gain["q_value_bh_across_21_contrasts"] = global_q.get(f"{endpoint}:{condition}")
    common = {
        "schema_version": 1, "scanner_order": list(frozen.SCANNER_ORDER),
        "cohorts": {
            "internal_test": {"subjects": 140, "bootstrap_n": len(samples["internal_test"]), "bootstrap_seed": SEEDS["internal_test"]},
            "onharmony": {"sessions": 165, "participants": 20, "participant_scanner_cells": 120,
                          "primary_acquisitions": 120, "scanners": 8, "scanners_per_participant": 6,
                          "bootstrap_n": len(samples["onharmony"]), "dbicc_bootstrap_seed": SEEDS["onharmony"],
                          "scanner_effect_bootstrap_seed": SEEDS["scanner_effects"]},
        },
        "accuracy_available": ages is not None,
    }
    return ({**common, "conditions": conditions_frozen}, {
        **common, "conditions": conditions_adapted,
        "gain_definitions": {"orientation": "positive values always indicate improvement after adaptation"},
        "training_hierarchy": {"aggregation": "per-run metrics averaged within shared bootstrap; predictions are never averaged"},
        "multiplicity": {"global_21_available": len(all_p) == 21,
                         "reason_if_unavailable": None if len(all_p) == 21 else "Internal-test ages were not provided"},
    })


def source_hashes(data_root: Path, ages: Path | None) -> dict:
    paths = [*data_root.rglob("*.csv"), data_root / "selections.json", data_root / "PRIMARY_PROTOCOL.json", Path(__file__).resolve(),
             Path(embedding.__file__), Path(frozen.__file__), Path(adaptation.__file__),
             ROOT / "FineTuning/shared/metrics.py", ROOT / "tools/prepare_reproduction_selections.py",
             embedding.R_SCRIPT]
    if ages is not None:
        paths.append(ages)
    return {str(path.resolve()): digest(path) for path in paths}


def write_summary(output: Path, representations: dict, frozen_result: dict, adapted_result: dict) -> None:
    rows = []
    def add(condition, arm, endpoint, record):
        if record is not None:
            rows.append({"condition": condition, "arm": arm, "endpoint": endpoint,
                         **{key: record.get(key) for key in ("estimate", "ci_lower", "ci_upper", "p_value_two_sided",
                             "q_value_bh", "q_value_bh_within_endpoint", "q_value_bh_across_21_contrasts")}})
    for condition, result in representations["conditions"].items():
        add(condition, "embedding", "dbicc", result["dbicc"])
    for condition, result in adapted_result["conditions"].items():
        for arm, metrics in result["absolute"].items():
            for name, record in metrics.items():
                add(condition, arm, name, record)
        for name, record in result["primary_gains"].items():
            add(condition, "adaptation_gain", name, record)
        add(condition, "prediction_minus_embedding", "dbicc", frozen_result["conditions"][condition]["dbicc_change"])
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, required=True, help="New output directory; existing outputs are not overwritten")
    parser.add_argument("--ages", type=Path, help="Private CSV with dataset, subject, age for the 140 internal-test subjects")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--reml", action="store_true", help="Also recompute crossed variance components using R/lme4")
    parser.add_argument("--workers", type=int, default=1, help="R worker processes")
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    data_root, output = args.data_root.expanduser().resolve(), args.output_root.expanduser().resolve()
    ages_path = args.ages.expanduser().resolve() if args.ages is not None else None
    require(args.n_bootstrap > 0 and args.workers > 0, "Bootstrap count and worker count must be positive")
    require(not output.is_relative_to(data_root), "Outputs must be outside the published data directory")
    if output.exists():
        raise FileExistsError(output)
    before = source_hashes(data_root, ages_path)
    verify(SimpleNamespace(data_root=data_root))
    cohort, acquisitions = read_csv(data_root / "cohort_splits.csv"), read_csv(data_root / "ON-Harmony/acquisitions.csv")
    ages = checked_ages(ages_path, cohort) if ages_path is not None else None
    samples = {key: embedding.bootstrap_indices(count, args.n_bootstrap, SEEDS[key])
               for key, count in (("onharmony", 20), ("internal_test", 140))}
    output.mkdir(parents=True)
    representations = representation_analysis(data_root, output, samples["onharmony"], args)
    frozen_result, adapted_result = prediction_analysis(data_root, acquisitions, cohort, representations, ages, samples)
    require(before == source_hashes(data_root, ages_path), "Input CSVs or analysis code changed during calculation")
    code = {"analysis_script": str(Path(__file__).resolve()), "analysis_script_sha256": digest(Path(__file__).resolve())}
    for name, payload in (("figure2_reliability_results.json", representations),
                          ("figure3_brain_age_results.json", frozen_result),
                          ("figure4_adaptation_results.json", adapted_result)):
        payload.update({"code": code, "created_utc": datetime.now(timezone.utc).isoformat(),
                        "input_mode": "published compact CSVs; no internal audit bundles reconstructed"})
        embedding.atomic_json(output / name, payload)
    write_summary(output, representations, frozen_result, adapted_result)
    audit = {"status": "PASS", "accuracy_available": ages is not None, "reml_calculated": args.reml,
             "global_21_contrasts_available": ages is not None, "bootstrap_n": args.n_bootstrap, "seeds": SEEDS,
             "inputs_and_code_sha256": before,
             "bootstrap_samples_sha256": {key: hashlib.sha256(value.astype(np.int64).tobytes()).hexdigest()
                                          for key, value in samples.items()},
             "outputs": {path.name: digest(path) for path in output.glob("*.json")}}
    audit["outputs"]["metrics.csv"] = digest(output / "metrics.csv")
    embedding.atomic_json(output / "reproduction_audit.json", audit)
    print(f"PASS: {output}; accuracy={'included' if ages is not None else 'unavailable (no ages provided)'}", flush=True)


if __name__ == "__main__":
    main()
