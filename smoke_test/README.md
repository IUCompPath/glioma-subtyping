# Smoke test: training → evaluation → multi-scale fusion

A few-minute end-to-end check of everything after feature extraction, on tiny pre-extracted features.
It needs no slides, no Hugging Face token and no downloads.

```bash
conda activate glioma_subtyping
smoke_test/run.sh                       # from the repository root; ends with "SMOKE TEST PASSED"
MODELS="att_mil" MAGS="10x 20x" smoke_test/run.sh   # quicker subset
```

## What it runs
For each backbone × model × magnification (2.5x, 5x, 10x, 20x) it trains 2 folds
(5 epochs), evaluates on the held-out test slides, then late-fuses all magnification
combinations (`pipeline/ensemble_script.py`), and finally verifies every output with `check.py`.
Models: `att_mil`, `mamba_mil` (needs the compiled Mamba kernels, see the main README), `clam_sb`.

## Data (`data/`)
| Backbone | Features | Source |
|---|---|---|
| `gigapath` (1536-d) | **real**, committed, ~5 MB | 12 TCGA slides (4 per WHO-2021 class, one per patient), 32 randomly sampled patches per slide at each magnification, from the full-size features. Built by `tools/build_smoke_features.py` |
| `uni` (1024-d) | **synthetic**, generated on first run (not committed) | `tools/make_synthetic_features.py`: class-dependent Gaussians |

`data/labels.csv` holds the slide labels; `data/splits/splits_{0,1}.csv` define two folds
(per class: 2 train / 1 val / 1 test slides, rotated between folds). Add other backbones
with `BACKBONES="gigapath uni virchow" smoke_test/run.sh` (synthetic features are created automatically).

> Only 3 test slides per fold, subsampled patches: the metrics are meaningless.
> This checks that the code runs and writes the right files, not that the models are good.
