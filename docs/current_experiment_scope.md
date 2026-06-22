# Current Experiment Data Protocol

This document describes the current **data protocol** only.

The broader research plan, research questions, baselines, and module roadmap remain preserved in:

```text
README.md
docs/original_research_plan.md
```

## What Changed

The active experiments are currently run on:

```text
data/exp/promise_exp.csv
```

This means the immediate experiment setup uses PROMISE expanded as the working dataset instead of the previously merged/capped corpus.

## What Did Not Change

The following broader research components are not removed:

- Research questions
- Baseline plan
- ModernBERT and ELECTRA ablations
- Hybrid model direction
- GPT-style LLM baseline direction
- Robustness/paraphrase research direction
- Overall paper plan

## Active Data File

- File: `data/exp/promise_exp.csv`
- Columns: `ProjectID`, `RequirementText`, `class`
- Labels: original PROMISE expanded labels
- Current adapted notebooks create train/validation/test splits at runtime.

## Note

This is a practical dataset-scope adjustment for running experiments. It is not a replacement for the broader research architecture or paper plan.
