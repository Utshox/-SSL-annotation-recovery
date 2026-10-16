# Annotation recovery of uncertainty-aware semi-supervised learning: code, splits and per-patient results

Code and data accompanying *How Much Annotation Value Does Uncertainty-Aware Semi-Supervised Learning Recover? A Matched-Control Measurement Across Label Budgets in Pancreas CT* (Ahmed and Komurcu, under review).

The repository contains everything needed to (a) retrain every corrected-protocol arm and (b) regenerate every corrected-protocol table and figure from the released per-patient scores without retraining. Model weights are not included because of their size.

## Layout

| Path | Contents |
|---|---|
| `baseline/code/` | Preprocessing, training, inference, scoring and analysis scripts |
| `baseline/models/<run>/fold_<k>/` | Per fold: `split_info.json` (exact patient IDs of the labeled, unlabeled, selection and scored sets), `dice_results_gaussian.json` (per-patient Dice), `best_info.json` (selected epoch), `log.csv` (training curve) |
| `baseline/logs/hd95_5fold*/` | Per-patient HD95, ASD and Dice, raw and after largest-component filtering (JSONL) |
| `baseline/logs/tcia_crossdomain/` | Per-patient Dice on the external NIH Pancreas-CT cohort, one line per volume and fold model (JSONL) |
| `baseline/logs/*.md, *.json` | Pooled results as reported in the paper |

## Data

* MSD Task07 Pancreas (281 training volumes): http://medicaldecathlon.com/ , place in `data/Task07_Pancreas/`.
* NIH Pancreas-CT (TCIA), 80 of 82 volumes (cases 25 and 70 are duplicates): https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU , converted to NIfTI in `data/tcia/imagesTs` and `data/tcia/labelsTs`.

Pancreas and tumour labels of MSD are merged into one foreground class. HU window [-125, 275].

## Runs

| Run name | Arm |
|---|---|
| `clean_sup_{10,25,50,100}` | Matched supervised control (corrected protocol). `clean_sup_10_es30` is the primary 10% control (early stopping from epoch 30); `clean_sup_10` stops from epoch 1 and is the rule-matched control for Mean Teacher, CPS and arm B at 10% |
| `clean_uamt_{10,25,50}` | UA-MT (soft Monte Carlo variance weighting) |
| `clean_sup_25_ts{1,2}`, `clean_uamt_25_ts{1,2}` | Training-seed repeats at 25% (seed 42 is the unsuffixed run) |
| `clean_mt_{10,50}`, `clean_cps_{10,50}` | Mean Teacher, cross-pseudo supervision |
| `clean_armb_{10,50}` | Step-matched arm B: consistency term on labeled images only |
| `aug_sup_10`, `aug_uamt_10` | Augmentation x SSL design, 10% (with `clean_sup_10_es30`, `clean_uamt_10`) |
| `augab_*` | Same design with an alternative augmentation set |

"Corrected protocol": checkpoints selected on 22 inner-validation patients per fold; unlabeled patches sampled from a body mask without opening label files (`preprocess_v7_labelfree.py`). Labeled patches come from `preprocess_v5_patches.py`.

## Reproducing the tables and figures from the released scores

Run from the repository root:

```bash
python baseline/code/collect_oct15.py            # matched contrasts, seeds, MT/CPS, arm B
python baseline/code/recompute_10pct_es30.py     # 10% numbers with the primary control
python baseline/code/label_equivalent_corrected.py
python baseline/code/collect_oct16_eval.py       # HD95/ASD/LCC and TCIA contrasts
python baseline/code/collect_aug_2x2.py
python baseline/code/make_tables_corrected.py    # LaTeX tables
python baseline/code/make_figs_corrected.py      # figures (writes overleaf_export/images/)
```

## Retraining

Examples (one fold; folds 0-4):

```bash
python baseline/code/run_supervised_fraction.py --data_dir data/preprocessed_v5_patches --raw_dir data/Task07_Pancreas \
  --fold 0 --frac 10 --inner_val 22 --blend gaussian --epochs 150 --output_dir baseline/models/clean_sup_10_es30
python baseline/code/run_ssl_uamt_5fold.py --data_dir data/preprocessed_v5_patches --raw_dir data/Task07_Pancreas \
  --unlabeled_data_dir data/preprocessed_v7_labelfree --fold 0 --frac 10 --inner_val 22 --blend gaussian --epochs 150 \
  --output_dir baseline/models/clean_uamt_10
```

See `--help` of each script for all options (including `--consistency_source labeled` for arm B, `--train_seed`, `--aug`).
Each run takes under three GPU-hours on a V100.

## Known limitation

The semi-supervised arms take two optimizer steps per control step within an epoch and sample batches with replacement; the control iterates a shuffled pass. See the paper's Limitations section.

## License

MIT (see `LICENSE`).
