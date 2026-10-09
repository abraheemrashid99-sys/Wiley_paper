#!/usr/bin/env bash
# Full pipeline for one config. Usage:
#   bash scripts/run_all.sh configs/dev.yaml [seeds] [methods]
# Stream runs that already have a result file are skipped, so this script can
# be re-started after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${1:?usage: run_all.sh <config.yaml> [seeds] [methods]}
SEEDS=${2:-0,1,2}
METHODS=${3:-frozen,tx_only,single_rate,ewc,nested,nested_no_ema,nested_two_level}
PY=${PYTHON:-python}

$PY scripts/check_env.py       --config "$CFG"
$PY scripts/prepare_data.py    --config "$CFG"
$PY scripts/benchmark_speed.py --config "$CFG"
$PY scripts/pretrain.py        --config "$CFG"
$PY scripts/eval_rate.py       --config "$CFG"
$PY scripts/run_stream.py      --config "$CFG" --methods "$METHODS" --seeds "$SEEDS"
$PY scripts/run_stream.py      --config "$CFG" --methods "$METHODS" --seeds "$SEEDS" stream.kind=temporal
$PY scripts/eval_refine.py     --config "$CFG"
$PY scripts/aggregate.py       --config "$CFG"
$PY scripts/make_figures.py    --config "$CFG"
echo
echo "DONE - tables in runs/*/tables, figures in runs/*/figures"
