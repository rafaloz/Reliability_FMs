"""Compact validated per-run exports into the study's results-only CSV layout."""

import argparse
import csv
import hashlib
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    expected = [{key: str(row.get(key, "")) for key in fields} for row in rows]
    assert read_csv(path)[1] == expected, path


def compact(source, output):
    if output.exists():
        raise FileExistsError(output)
    temporary = Path(tempfile.mkdtemp(prefix="compact_", dir=output.parent))
    try:
        shutil.copy2(source / "sources.csv", temporary / "sources.csv")
        total = 0
        for dataset in sorted(p for p in source.iterdir() if p.is_dir()):
            _, inventory = read_csv(dataset / "inventory.csv")
            groups = defaultdict(list)
            for entry in inventory:
                path = Path(entry["file"])
                if dataset.name not in {"CamCAN", "ON-Harmony"}:
                    if entry["arm"] == "embedding" or entry["split"] != "internal_test":
                        continue
                target = (f"embeddings/{path.parts[1]}.csv" if entry["arm"] == "embedding"
                          else "predictions.csv")
                groups[target].append(entry)
            destination = temporary / dataset.name
            destination.mkdir()
            provenance = []
            for target, entries in sorted(groups.items()):
                fields, rows = [], []
                feature_sets = set()
                for entry in entries:
                    original = dataset / entry["file"]
                    assert digest(original) == entry["csv_sha256"], original
                    columns, records = read_csv(original)
                    assert len(records) == int(entry["rows"])
                    feature_sets.add(tuple(c for c in columns if c.startswith("emb_")))
                    fields.extend(c for c in columns if c not in fields)
                    rows.extend(records)
                assert len(feature_sets) == 1, "Cannot mix incompatible embedding dimensions"
                write_csv(destination / target, fields, rows)
                checksum = digest(destination / target)
                for entry in entries:
                    provenance.append({**entry, "source_file": entry["file"],
                                       "source_csv_sha256": entry["csv_sha256"],
                                       "file": target, "csv_sha256": checksum})
                total += 1
            columns = list(dict.fromkeys(key for row in provenance for key in row))
            write_csv(destination / "inventory.csv", columns, provenance)
            shutil.copy2(dataset / "data_dictionary.csv", destination / "data_dictionary.csv")
            notes_path = dataset / "deposit_notes.csv"
            if notes_path.exists():
                columns, notes = read_csv(notes_path)
                for note in notes:
                    if note["item"] == "scope":
                        note["description"] = ("All CamCAN derived rows, consolidated by model and output type."
                            if dataset.name == "CamCAN" else "Internal-test predictions for all retained runs.")
                write_csv(destination / "deposit_notes.csv", columns, notes)
        assert total == 22, total
        temporary.rename(output)
        print(f"PASS: {total} data CSVs; all retained values round-trip unchanged")
    except BaseException:
        shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compact(args.source, args.output)
