# Study selections

These tables identify the exact subjects, splits and acquisitions used in the
study. They contain no ages, sex, image paths or predicted values.

| File | Columns | Purpose |
| --- | --- | --- |
| `cohort_splits.csv` | `dataset`, `subject`, `split` | All 1,411 development subjects, in the original manifest order. Split sizes: 1,130 train, 141 validation, 140 internal test. |
| `ON-Harmony/acquisitions.csv` | `participant_id`, `session_id`, `scanner`, `phase`, `primary_acquisition` | All 165 scans. `True` marks the first scan in each participant-scanner cell: 120 scans, six per participant. |
| `selections.json` | SHA-256 values, column names and row counts | Identity of the source records and the two published selection tables. |

IDs are the original dataset identifiers. Join development demographics using
both `dataset` and `subject`; IDs alone need not be unique across datasets.
Published prediction CSVs call the subject column `participant_id`. Use
`participant_id` and `session_id` together for ON-Harmony acquisitions.
`phase` identifies the two ON-Harmony acquisition phases, not the participant's
clinical status. Get images and demographics from the dataset owners using
the links in [`sources.csv`](sources.csv).

```bash
python tools/prepare_reproduction_selections.py verify --data-root data
```

This checks the split counts, primary-acquisition selection, phase coverage,
and agreement with every published embedding/prediction CSV, including
separate model runs and folds. CSV contents are checked against their existing
dataset inventories.

To regenerate the tables from the original study records:

```bash
python tools/prepare_reproduction_selections.py export \
  --manifest /path/to/brainage_definitive_subject_manifest.tsv \
  --participants /path/to/onharmony/participants.tsv \
  --data-root data \
  --output /path/to/new-selection-export
```

`cohort_splits.csv` preserves selection, not the full internal training
manifest: it omits demographics, image paths and input contracts. It is not
a replacement for that manifest in the existing training commands. The
original hash checks in those commands are unchanged.
