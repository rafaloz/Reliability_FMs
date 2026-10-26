"""Export minimal study selections, or verify them against the published CSVs."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


MANIFEST_SHA256 = "b2ee2dc7e06349ef1ac550da2061274ee724d2b121ddd652b37c1907ae8bc212"
COHORT_FIELDS = ("dataset", "subject", "split")
ACQUISITION_FIELDS = (
    "participant_id", "session_id", "scanner", "phase", "primary_acquisition",
)
SPLITS = ("train", "validation", "internal_test")
COUNTS = {
    "CamCAN": (521, 65, 64),
    "MR-ART": (118, 15, 15),
    "NIMH-RV": (118, 14, 14),
    "NeuroCog": (239, 30, 30),
    "WAND": (134, 17, 17),
}
PHASE_SCANNERS = {
    "A": {"NOT1ACH", "NOT2ING", "NOT3GEM", "OXF1PRI", "OXF2PRI", "OXF3TRI"},
    "B": {"NOT1ACH", "NOT2ING", "NOT4GEP", "OXF1PRI", "OXF2PRI", "OXF4GEP"},
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected_rows(path, fields, *, delimiter=","):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        require(set(fields).issubset(reader.fieldnames or ()), f"Missing columns: {path}")
        return [{field: row[field] for field in fields} for row in reader]


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    require(selected_rows(path, fields) == rows, f"CSV round-trip mismatch: {path}")


def validate_cohort(rows):
    keys = [(row["dataset"], row["subject"]) for row in rows]
    require(len(keys) == len(set(keys)) == 1411, "Expected 1411 unique development subjects")
    require(all(row["subject"] for row in rows), "Empty development subject identifier")
    expected = Counter({(dataset, split): n for dataset, counts in COUNTS.items()
                        for split, n in zip(SPLITS, counts)})
    require(Counter((row["dataset"], row["split"]) for row in rows) == expected,
            "Development dataset/split counts do not match the study")
    return {split: sum(row["split"] == split for row in rows) for split in SPLITS}


def validate_acquisitions(rows):
    keys = [(row["participant_id"], row["session_id"]) for row in rows]
    require(len(keys) == len(set(keys)) == 165, "Expected 165 unique ON-Harmony acquisitions")
    phases, cells, participant_phases = defaultdict(set), defaultdict(list), {}
    for row in rows:
        participant, session, scanner, phase, primary = (row[key] for key in ACQUISITION_FIELDS)
        require(participant.startswith("sub-"), "Invalid ON-Harmony participant identifier")
        require(phase in PHASE_SCANNERS and scanner in PHASE_SCANNERS[phase],
                "Invalid phase/scanner combination")
        require(session.startswith(f"ses-{scanner}") and session[-3:].isdigit()
                and len(session) == len(f"ses-{scanner}") + 3, "Invalid ON-Harmony session")
        require(primary in {"True", "False"} and (primary == "True") == session.endswith("001"),
                "Primary acquisition must be session 001")
        require(participant_phases.setdefault(participant, phase) == phase,
                "ON-Harmony participant occurs in more than one phase")
        phases[phase].add(participant)
        cells[(participant, scanner)].append(row)
    require({phase: len(ids) for phase, ids in phases.items()} == {"A": 10, "B": 10},
            "Expected ten ON-Harmony participants in each phase")
    require(len(cells) == 120 and Counter(len(cell) for cell in cells.values()) == {1: 111, 6: 9},
            "Unexpected ON-Harmony repeat structure")
    for cell in cells.values():
        require(sum(row["primary_acquisition"] == "True" for row in cell) == 1,
                "Each participant/scanner cell must have exactly one primary acquisition")
    for participant, phase in participant_phases.items():
        require({scanner for subject, scanner in cells if subject == participant} == PHASE_SCANNERS[phase],
                "Incomplete scanner coverage for an ON-Harmony participant")
    return {"acquisitions": 165, "participants": 20, "primary_acquisitions": 120}


def verify_published(data_root, cohort, acquisitions):
    expected_subjects = defaultdict(set)
    for row in cohort:
        expected_subjects[(row["dataset"], row["split"])].add(row["subject"])
    acquisition_map = {
        (row["participant_id"], row["session_id"]):
        (row["scanner"], row["primary_acquisition"]) for row in acquisitions
    }
    checked = set()
    for dataset in (*COUNTS, "ON-Harmony"):
        folder = data_root / dataset
        inventory = selected_rows(folder / "inventory.csv", ("file", "csv_sha256"))
        for entry in inventory:
            path = (folder / entry["file"]).resolve()
            require(path.is_relative_to(folder.resolve()), "Inventory path leaves dataset directory")
            if path not in checked:
                checksum = digest(path)
                require(all(record["csv_sha256"] == checksum for record in inventory
                            if record["file"] == entry["file"]), f"Published CSV checksum mismatch: {path}")
                checked.add(path)
        if dataset == "ON-Harmony":
            for path in sorted(folder.glob("embeddings/*.csv")) + [folder / "predictions.csv"]:
                require(path.resolve() in checked, f"Uninventoried published CSV: {path}")
                fields = ("participant_id", "session_id", "scanner", "primary_acquisition",
                          "model", "variant", "arm", "run")
                groups = defaultdict(dict)
                for row in selected_rows(path, fields):
                    group = tuple(row[key] for key in ("model", "variant", "arm", "run"))
                    key = (row["participant_id"], row["session_id"])
                    require(key not in groups[group], f"Duplicate acquisition in {path}: {group}")
                    groups[group][key] = (row["scanner"], row["primary_acquisition"])
                require(groups and all(group == acquisition_map for group in groups.values()),
                        f"Published ON-Harmony identities/selection disagree: {path}")
        else:
            paths = [folder / "predictions.csv"] + sorted(folder.glob("embeddings/*.csv"))
            for path in paths:
                require(path.resolve() in checked, f"Uninventoried published CSV: {path}")
                fields = ("participant_id", "split", "model", "variant", "arm", "run")
                groups = defaultdict(list)
                for row in selected_rows(path, fields):
                    group = tuple(row[key] for key in ("model", "variant", "arm", "run", "split"))
                    groups[group].append(row["participant_id"])
                require(bool(groups), f"Empty published CSV: {path}")
                for group, subjects in groups.items():
                    require(len(subjects) == len(set(subjects))
                            and set(subjects) == expected_subjects[(dataset, group[-1])],
                            f"Published development identities/splits disagree: {path}: {group}")
    require(len(checked) == 22, f"Expected 22 published data CSVs, found {len(checked)}")
    return len(checked)


def export(args):
    require(digest(args.manifest) == MANIFEST_SHA256, "Wrong definitive development manifest SHA-256")
    cohort = selected_rows(args.manifest, COHORT_FIELDS, delimiter="\t")
    validate_cohort(cohort)
    phase_rows = selected_rows(args.participants, ("participant_id", "phase"), delimiter="\t")
    phases = {row["participant_id"]: row["phase"] for row in phase_rows}
    require(len(phases) == len(phase_rows) == 20, "Expected 20 unique ON-Harmony participants")
    metadata_fields = tuple(key for key in ACQUISITION_FIELDS if key != "phase")
    metadata_path = args.data_root / "ON-Harmony/embeddings/yaware.csv"
    acquisitions = []
    for row in selected_rows(metadata_path, metadata_fields):
        require(row["participant_id"] in phases, "Missing ON-Harmony phase")
        acquisitions.append({key: phases[row["participant_id"]] if key == "phase" else row[key]
                             for key in ACQUISITION_FIELDS})
    acquisitions.sort(key=lambda row: (row["participant_id"], row["session_id"]))
    validate_acquisitions(acquisitions)
    verify_published(args.data_root, cohort, acquisitions)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".selections_", dir=args.output.parent))
    try:
        outputs = {
            "cohort_splits.csv": (COHORT_FIELDS, cohort),
            "ON-Harmony/acquisitions.csv": (ACQUISITION_FIELDS, acquisitions),
        }
        for relative, (fields, rows) in outputs.items():
            write_csv(temporary / relative, fields, rows)
        provenance = {
            "schema": "reproduction_selections_v1",
            "sources": {
                "definitive_manifest_sha256": MANIFEST_SHA256,
                "onharmony_participants_sha256": digest(args.participants),
                "onharmony_metadata_csv": "ON-Harmony/embeddings/yaware.csv",
                "onharmony_metadata_sha256": digest(metadata_path),
            },
            "files": {name: {"sha256": digest(temporary / name), "rows": len(rows),
                             "columns": list(fields)} for name, (fields, rows) in outputs.items()},
        }
        (temporary / "selections.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        temporary.rename(args.output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(f"PASS: minimal selections exported to {args.output}")


def verify(args):
    metadata = json.loads((args.data_root / "selections.json").read_text())
    expected = {"cohort_splits.csv": COHORT_FIELDS, "ON-Harmony/acquisitions.csv": ACQUISITION_FIELDS}
    require(metadata["schema"] == "reproduction_selections_v1" and set(metadata["files"]) == set(expected),
            "Unexpected selection schema")
    require(metadata["sources"]["definitive_manifest_sha256"] == MANIFEST_SHA256,
            "Wrong source manifest identity")
    loaded = {}
    for relative, fields in expected.items():
        path = args.data_root / relative
        record = metadata["files"][relative]
        require(digest(path) == record["sha256"], f"Selection checksum mismatch: {path}")
        with path.open(newline="", encoding="utf-8") as stream:
            require(tuple(next(csv.reader(stream))) == fields, f"Unexpected selection columns: {path}")
        rows = selected_rows(path, fields)
        require(len(rows) == record["rows"] and list(fields) == record["columns"],
                f"Selection schema/count mismatch: {path}")
        loaded[relative] = rows
    sources = metadata["sources"]
    require(sources["onharmony_metadata_csv"] == "ON-Harmony/embeddings/yaware.csv"
            and digest(args.data_root / sources["onharmony_metadata_csv"]) == sources["onharmony_metadata_sha256"],
            "ON-Harmony metadata source changed")
    splits = validate_cohort(loaded["cohort_splits.csv"])
    onharmony = validate_acquisitions(loaded["ON-Harmony/acquisitions.csv"])
    count = verify_published(args.data_root, loaded["cohort_splits.csv"], loaded["ON-Harmony/acquisitions.csv"])
    print(json.dumps({"status": "PASS", "development_splits": splits,
                      "onharmony": onharmony, "published_data_csvs_checked": count}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser("export", help="Create selections from the original study records")
    export_parser.add_argument("--manifest", type=Path, required=True)
    export_parser.add_argument("--participants", type=Path, required=True)
    export_parser.add_argument("--data-root", type=Path, default=Path("data"))
    export_parser.add_argument("--output", type=Path, required=True, help="New output directory")
    export_parser.set_defaults(function=export)
    verify_parser = commands.add_parser("verify", help="Check published selections and CSV identities")
    verify_parser.add_argument("--data-root", type=Path, default=Path("data"))
    verify_parser.set_defaults(function=verify)
    arguments = parser.parse_args()
    arguments.function(arguments)
