#!/usr/bin/env bash

set -euo pipefail

CONFIG="$1"
GPUS="$2"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --nproc_per_node="$GPUS" \
    --master_port="$MASTER_PORT" \
    "$SCRIPT_DIR/train.py" \
    "$CONFIG" \
    --launcher pytorch "${@:3}"
