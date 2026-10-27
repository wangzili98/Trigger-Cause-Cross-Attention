# Trigger–Cause Cross-Attention (TCA)

PyTorch implementation of **Trigger-Cause Cross-Attention improves debris-flow early warning by learning state-dependent thresholds**. This repository provides TCA pretraining, supervised training, prediction and a minimum example with four real samples.

## Run

Use Python 3.12. Install the dependencies and run from this directory:

```bash
python -m pip install -r requirements.txt
python -B train/train_tca.py
python -B train/run_tca.py
```

Training performs one masked-reconstruction update for each meteorological encoder, followed by one supervised update on the four example samples. Prediction loads the resulting checkpoint and writes sample-level outputs. All model layers and embedding widths are retained in the minimum example.

CPU and CUDA are supported and selected automatically. For an explicit device or additional updates:

```bash
python -B train/train_tca.py --device cpu --epochs 10 --pretrain-steps 5
python -B train/run_tca.py --device cpu
```

`--data-dir PATH` selects another directory with the same input files; checkpoints and outputs are saved under that directory. Default paths are resolved relative to this repository. Scripts also work when invoked by absolute path from another working directory.

To pretrain the encoders separately and then fine-tune them:

```bash
python -B train/pretrain_tca.py --steps 1
python -B train/train_tca.py --encoder-checkpoint data/checkpoints/TCA_encoders.pt
```

## Model and objective

The Cause encoder uses four CNN blocks and a six-layer temporal Transformer. A separate CNN produces a 64-dimensional environmental representation. An MLP combines each dynamic Cause token with that environmental representation before generating the Query. The Trigger encoder uses two CNN blocks and a six-layer temporal Transformer to generate the Key and Value inputs.

Both temporal encoders use 256-dimensional embeddings and eight attention heads. Cross-attention uses eight heads, dropout 0.2, residual connections, layer normalization and a 1024-unit feed-forward block. Masked mean and maximum pooling feed the occurrence and auxiliary magnitude heads. Each head has a 128-unit hidden layer, GELU and dropout 0.4. The occurrence bias is initialized with a positive prior of 0.15.

Meteorological pretraining reconstructs 30% masked spatio-temporal patches without event labels. Supervised training jointly optimizes the encoders, environmental encoder, fusion and heads using AdamW and cosine annealing. Its objective is focal classification loss plus positive-only MSE between predicted log magnitude and `log1p(observed magnitude)`.

`config/tca_config.py` contains all network and optimization settings. The default focal alpha/gamma (0.9/2), regression weight (0.1) and optimizer group rates are carried over from the supplied training code; the manuscript does not report their final selected numerical values. One update and four samples are minimum-example settings.

## Input files

All inputs are under `data/`. Load NPZ files with `allow_pickle=False`.

| File | Arrays and shapes |
|---|---|
| `dynamic_data.npz` | `cause`: `(4,3,48,64,64)`; `trigger`: `(4,3,18,16,16)`; `cause_valid`: `(4,48)`; `trigger_valid`: `(4,18)` |
| `static_data.npz` | `environment`: `(4,16,64,64)` |
| `target_data.npz` | `occurrence`: `(4,)`; original nonnegative `magnitude`: `(4,)` |

Each file includes matching `sample_ids`. Frame availability is boolean, with true indicating available data. The dynamic and static archives include their feature metadata. Rainfall values are already log-transformed in the supplied inputs; the loader does not apply that transformation again. Magnitude is transformed once inside the training loss.

Normalization is fitted on the example training inputs and saved in the checkpoint. Prediction uses the saved statistics and does not require targets. Missing frames are excluded from encoding, attention and pooling. A completely unavailable Trigger sequence contributes zero attention.

The four records comprise two positives and two negatives from May 2025. Source filenames, row indices and hashes are recorded in `data/provenance.json`. These samples demonstrate training and input–output behavior; they are reused for prediction and do not form an independent evaluation set. The example does not reproduce the paper's full 2020–2024 fitting and independent 2025 evaluation.

To extract the same subset from the original sample directory:

```bash
python -B scripts/prepare_example.py --source-dir /path/to/fusion
```

## Output files

- `data/checkpoints/TCA.pt`: weights, architecture, optimization settings, normalization, data hashes and training histories.
- `data/training_history.csv`: combined loss, focal loss and positive-only log-magnitude MSE.
- `data/outputs/predictions.csv`: sample IDs, probabilities, log magnitudes and inverse-transformed magnitudes.
- `data/outputs/TCA_outputs.npz`: predictions, Cause/Trigger tokens, projected Query/Key/Value, availability and per-head attention.
- `data/outputs/metadata.json`: shapes, axes and transformations.

Attention has shape `(4,8,48,18)` in the example, ordered as sample, head, Cause time and Trigger time. Query has shape `(4,48,256)` and Key/Value have shape `(4,18,256)`. Invalid attention rows and columns are zero. Auxiliary magnitude is exported as `expm1(clip(predicted_log_magnitude,0,20))`, together with its raw log prediction.

Checkpoints and generated outputs are excluded from Git; the commands above generate them locally.

## Tests

```bash
python -B -m unittest discover -s tests -v
```

The tests cover the real-data contract, Query conditioning by environmental features, missing-frame masking, all-missing inputs, the regression objective, pretraining masks and checkpoint reload.

## Citation and license

Software authors: Zili Wang and Peng Cui. Cite the archived version using `CITATION.cff`. The intended Zenodo record is https://doi.org/10.5281/zenodo.20580569; confirm that it contains the released code before citing it. `.zenodo.json` supplies the GitHub–Zenodo metadata. This local working copy has not been published by this task.

The software is distributed under the MIT license. See `data/DATA_NOTICE.md` for the sample-data notice.
