# Pair-Representation-Autoencoder
A Pairformer-based autoencoder for compressing interchain pair representations produced by AlphaFold 3 (AF3) or RoseTTAFold 3 (RF3) trunks.

Developed by Evan Meltzer as an undergraduate research project in the Baker Lab at the University of Washington.

## Overview

AF3- and RF3 trunks produce large two-dimensional pair representations. Storing these tensors is expensive, and passing them directly to downstream models can add substantial memory and compute costs. This project compresses the channel dimension of interchain pair features.

The included configuration compresses pair features from 128 channels to 16 channels and then reconstructs them back to 128 channels.

## Architecture

The autoencoder uses Pairformer stages which are optimized to operate over the rectangular shape of interchain regions.

Input: Interchain pair representation (`L1 x L2 x 128`)
Encoder: Consists of 4 Pairformer stages, reducing the channels as `128 -> 64 -> 32 -> 16`
Bottleneck: Compressed pair representation (`L1 x L2 x 16`)
Decoder: Consists of 4 Pairformer stages, increasing the channels as `16 -> 32 -> 64 -> 128`
Ouput: Reconstructed interchain pair representation (`L1 x L2 x 128`)

Training uses three objectives:
- Mean-squared reconstruction loss
- Cosine reconstruction loss
- A distance-consistency loss from a frozen distogram head

## Data format

The training and validation datasets are CSV files with a `path` column. Each path identifies an `.npz` file containing:

```text
pair_embeddings: float array with shape (L1, L2, 128)
```

Intrachain regions must be zeroed before the data are passed to the model, which treats positions whose feature channels are all zero as masked cells.

## Required model assets

Two additional files are required:

- Normalization statistics: a PyTorch checkpoint containing `mean` and `std` tensors, each with shape `(128,)`
- Frozen distogram-head weights: weights acquired from a pretrained frozen distogram head compatible with the 128-channel pair representation

## Installation

Python 3.10 or newer is required.

```bash
git clone <repository-url>
cd pair-representation-autoencoder
pip install -e .
```

## Configuration

The main experiment configuration is:

```text
compressor/configs/compressor.py
```

It contains the hyperparameters used for the best-performing experiments, including:

- Bottleneck dimension: `16`
- Encoder dimensions: `128 -> 64 -> 32 -> 16`
- Decoder dimensions: `16 -> 32 -> 64 -> 128`
- Four transformer blocks per stage
- Four attention heads
- Equal weighting of MSE, cosine, and distogram losses

Before training, set the required paths:

```bash
export DISTOGRAM_HEAD_WEIGHT_PATH=/path/to/distogram_head_weights.pt
export NORMALIZATION_STATS_PATH=/path/to/normalization_stats.pt
export TRAIN_ANN_FILE=/path/to/train.csv
export VAL_ANN_FILE=/path/to/validation.csv
```

Optional dataset roots can be supplied when the CSV paths are relative:

```bash
export TRAIN_DATA_ROOT=/path/to/training/data
export VAL_DATA_ROOT=/path/to/validation/data
```

## Training

Training commands depend on the compute cluster. The repository includes an MMEngine entry point, a `torchrun` launcher, and an example Slurm script.

### Distributed training with `torchrun`

```bash
bash tools/dist_train.sh compressor/configs/compressor.py <number-of-gpus>
```

For example:

```bash
bash tools/dist_train.sh compressor/configs/compressor.py 8
```

An alternate output directory can be provided through the training script:

```bash
bash tools/dist_train.sh compressor/configs/compressor.py 8 \
    --work-dir output/experiment_name
```

### Slurm

Set the Conda environment, required data paths, and model assets before submitting:

```bash
export CONDA_ENV=<conda-environment-name-or-path>
export DISTOGRAM_HEAD_WEIGHT_PATH=/path/to/distogram_head_weights.pt
export NORMALIZATION_STATS_PATH=/path/to/normalization_stats.pt
export TRAIN_ANN_FILE=/path/to/train.csv
export VAL_ANN_FILE=/path/to/validation.csv

sbatch train.sbatch
```

## Outputs

By default, training writes to:

```text
output/debug/
```

The output directory contains MMEngine logs, TensorBoard event files, and model checkpoints. Checkpoints are saved every 5,000 iterations, with up to three retained by the default configuration. Validation also runs every 5,000 iterations.

The evaluator reports:

- `compressor/z_ii_l1`: reconstruction L1 error
- `compressor/z_ii_cosine_sim`: reconstruction cosine similarity
- `compressor/distogram_l1`: L1 difference between expected distances from the original and reconstructed representations

## Results

The best logged run used a 16-channel bottleneck, reducing each 128-channel pair representation by 8x. Validation was performed every 5,000 training iterations on 576 held-out examples.

| Metric | Best training value | Best validation value |
|---|---:|---:|
| Cosine similarity | 0.9483 | 0.925 |
| Distogram L1 | 0.0311 | 0.0459 |

## Limitations

- The current implementation supports a batch size of one. Pair representations are large enough that larger batches exceeded available GPU memory during development.

## Repository structure

```text
compressor/
  configs/       MMEngine training configuration
  datasets/      CSV/NPZ loading, transforms, and evaluation
  models/        Row/column autoencoder and distogram objective

tools/
  train.py       MMEngine training entry point
  dist_train.sh  Distributed torchrun launcher

train.sbatch     Example Slurm submission script
pyproject.toml   Package and dependency metadata
```
