#!/usr/bin/env bash
#SBATCH --job-name=pair-compressor
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/compressor/configs/compressor.py}"
NUM_GPUS="${NUM_GPUS:-8}"
CONDA_ENV="${CONDA_ENV:?Set CONDA_ENV to a Conda environment name or path}"

TMPDIR="${TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 20000))}"
export TORCH_HOME="${TORCH_HOME:-$TMPDIR/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$TMPDIR/triton}"
mkdir -p "$TORCH_HOME" "$TRITON_CACHE_DIR"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: conda not found in PATH" >&2
    exit 1
fi

conda activate "$CONDA_ENV"
bash "$REPO_ROOT/tools/dist_train.sh" "$CONFIG_PATH" "$NUM_GPUS"
