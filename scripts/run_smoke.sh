#!/usr/bin/env bash
# Offline end-to-end check on a tiny model and toy text (a few minutes on a laptop).
# Verifies that every stage runs; the numbers it prints mean nothing scientifically.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=configs/smoke.yaml
PY=${PYTHON:-python}

$PY scripts/check_env.py       --config $CFG
$PY scripts/prepare_data.py    --config $CFG
$PY scripts/pretrain.py        --config $CFG
$PY scripts/eval_rate.py       --config $CFG
$PY scripts/run_stream.py      --config $CFG --methods frozen,tx_only,single_rate,nested --seeds 0,1
$PY scripts/eval_refine.py     --config $CFG
$PY scripts/aggregate.py       --config $CFG
$PY scripts/make_figures.py    --config $CFG
echo
echo "SMOKE TEST PASSED - outputs in runs/smoke/"
