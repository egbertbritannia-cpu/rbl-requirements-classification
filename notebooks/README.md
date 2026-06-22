# Notebooks

Active notebooks are configured to use:

```text
../data/exp/promise_exp.csv
```

## Active Baselines

- `EDA-promise_exp.ipynb`: quick dataset inspection.
- `SVM.ipynb`: adapted SVM baseline using a stratified 70/15/15 split from `promise_exp.csv`.
- `ELECTRA.ipynb`: adapted ELECTRA baseline using `google/electra-base-discriminator` by default.
- `Hybrid_BERT_RoBERTa.ipynb`: adapted hybrid baseline using a stratified 70/15/15 split from `promise_exp.csv`.

## Origin Notebooks

The `*_origin.ipynb` files are kept close to the author's original notebooks:

- `Bert_origin.ipynb`
- `RoBerta_origin.ipynb`
- `Hybrid_BERT_RoBERTa_origin.ipynb`
- `Hybrid_BERT_RoBERTa_without_augmentation_origin.ipynb`

For these origin notebooks, only the data-loading path has been changed so they read `data/exp/promise_exp.csv`. Core preprocessing, augmentation, split, architecture, and training logic are intentionally preserved.
