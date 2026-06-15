# Datasets

This directory contains all raw, processed, and augmented datasets used in the project.

## Directory Structure

```
data/
├── raw/                    # Original, unmodified source datasets
├── processed/              # Cleaned, merged, and split data (train/val/test)
└── paraphrased/            # LLM-generated paraphrase variants (for RQ2)
```

## Raw Datasets

### 1. PROMISE (Relabeled) — Primary Source
- **File**: `raw/PROMISE-relabeled-NICE.csv`
- **Records**: 622
- **Columns**: `ProjectID`, `RequirementText`, `IsFunctional`, `IsQuality`, + 12 NFR subtype columns (binary 0/1)
- **Source**: [PROMISE Software Engineering Repository](http://openscience.us/repo/requirements/nfr.html), relabeled by NICE framework
- **Mapping**: `IsFunctional=1` → FR, else → NFR

### 2. PROMISE (Expanded)
- **File**: `raw/promise_exp.csv`
- **Records**: 969
- **Columns**: `ProjectID`, `RequirementText`, `class`
- **Label values**: F, SE, US, PE, LF, O, A, MN, SC, FT, L, PO
- **Mapping**: `class='F'` → FR, else → NFR
- **Note**: Deduplicated against PROMISE-relabeled-NICE (primary source takes priority)

### 3. DCAI24 (Merged Multi-Source)
- **File**: `raw/dcai24_src_dataset.xlsx`
- **Records**: 3,482
- **Columns**: `Requirement`, `Type`, `Specific_Type`, `Security_Category`, `Dataset_Name`
- **Mapping**: `Type='FR'` → FR, `Type='NFR'` → NFR (direct mapping)

### 4. EARS (Functional Requirements Only)
- **File**: `raw/EARS Functional Requirements Complete Dataset.xlsx`
- **Records**: 9,677
- **Columns**: `Projects`, `Raw Requirements`, `Requiremnet Name`, `EARS Type`, `EARS Requirement`
- **EARS Types**: ubiquitous, state driven, event driven, unwanted behaviour, optional feature
- **⚠️ Important**: This dataset contains **only FR** samples. All records are mapped to FR. Used selectively for class imbalance handling.
- **Known quality issues**: Inconsistent casing (`Ubiquitous` vs `ubiquitous`), leading whitespace in labels, truncated labels (`unwanted` vs `unwanted behaviour`). Cleaning is required before use.

## Label Unification

All datasets are merged into a unified binary classification task:

| Label | Value | Description |
|-------|-------|-------------|
| FR | 0 | Functional Requirement |
| NFR | 1 | Non-Functional Requirement |

## Data Splits

After merging and deduplication, data is split into:
- **Train**: 80%
- **Validation**: 10%
- **Test**: 10%

Stratified splitting is used to preserve class distribution across splits.