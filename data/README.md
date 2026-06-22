# Data

The active dataset for the current paper scope is:

```text
data/exp/promise_exp.csv
```

## Active Dataset

### PROMISE Expanded

- File: `exp/promise_exp.csv`
- Rows: 969
- Columns: `ProjectID`, `RequirementText`, `class`
- Labels in file: `F`, `SE`, `US`, `PE`, `LF`, `O`, `A`, `MN`, `SC`, `FT`, `L`, `PO`

The notebooks use this file directly. Adapted notebooks create a stratified 70/15/15 train/validation/test split at runtime.

## Directory Meaning

```text
data/
├── exp/                 # Active experiment dataset
├── raw/                 # Source/archive files
├── processed/           # Legacy merged-corpus split, not active
├── processed_ablation/  # Legacy ablation split, not active
└── paraphrased/         # Optional future robustness data
```

## Current Protocol

The current repository version does not use the previous merged FR-heavy corpus. It uses `promise_exp.csv` as the single source of data for the paper experiments.

The `class` column is preserved as provided by PROMISE expanded. No external FR-only dataset is merged into this protocol.
