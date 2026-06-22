# Current Research Scope

The current repository version uses only:

```text
data/exp/promise_exp.csv
```

The previous merged FR-heavy corpus is no longer the active paper protocol.

## Research Questions

1. How do classical and transformer-based baselines perform on the PROMISE expanded multi-class requirements classification task?
2. How does preprocessing affect BERT/RoBERTa-style baselines on PROMISE expanded?
3. How does augmentation affect minority classes in the PROMISE expanded setup?
4. Does a BERT + RoBERTa hybrid architecture improve performance over single-model baselines under the same dataset protocol?

## Experimental Boundary

- Dataset: PROMISE expanded (`promise_exp.csv`)
- Labels: original `class` values from the file
- Active split: stratified 70/15/15
- No external FR-only dataset is merged
- Legacy `data/processed*` files are not part of the active protocol
