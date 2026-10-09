# Doubly Nested Semantic Communication (DN-SC)

Research code for a rate- and time-nested LLM semantic codec for drift-resilient
6G links, prepared for the *International Journal of Communication Systems*
special issue "Next-Generation 6G Mobile Networks and Services: LLM-Enabled
Semantic Communications" (submission deadline 31 March 2027).

**Status: v0.1 — experimental pipeline complete and tested; no scientific
result has been produced yet.** The code has been verified end to end on a tiny
offline model and on two real corpora. The first run with a pretrained LLM
happens on your machine (see "What has and has not been verified").

## The idea in four lines

- **Nested in rate.** One encoder emits `L` channel symbols; *any prefix* is a
  valid transmission. The payload adapts to the link, not the model.
- **Nested in time.** Transceiver parameters are split into levels updated at
  separated frequencies. Receiver-side levels must be synchronised over the
  air, and every synchronisation bit is counted.
- **Net semantic rate.** Payload symbols plus amortised synchronisation
  symbols per message — the quantity that decides whether adapting a learned
  codec to a drifting link pays for itself, and at what traffic volume.
- **Calibrated refinement.** The receiver requests the next shell of symbols
  only when its calibrated confidence is low: semantic incremental redundancy.

Details: [`docs/METHOD.md`](docs/METHOD.md) · datasets:
[`docs/DATASETS.md`](docs/DATASETS.md) · 15-day plan:
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## Quick start (macOS, Apple Silicon)

Python 3.10 or newer is required (`python3 --version`; if it is older,
`brew install python@3.12` and use `python3.12` below).

```bash
git clone https://github.com/abraheemrashid99-sys/Wiley_paper.git
cd Wiley_paper
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

pytest                      # unit + integration tests, offline, a few minutes
bash scripts/run_smoke.sh   # whole pipeline on a tiny offline model
```

Then the first real run, on the small development backbone:

```bash
python scripts/check_env.py       --config configs/dev.yaml   # downloads model + data, one step on MPS
python scripts/prepare_data.py    --config configs/dev.yaml   # builds and caches the four corpora
python scripts/benchmark_speed.py --config configs/dev.yaml   # it/s and time estimates on this machine
python scripts/pretrain.py        --config configs/dev.yaml   # base codec
python scripts/eval_rate.py       --config configs/dev.yaml   # rate-fidelity + separation references
python scripts/run_stream.py      --config configs/dev.yaml --methods frozen,tx_only,single_rate,nested --seeds 0,1,2
python scripts/eval_refine.py     --config configs/dev.yaml
python scripts/aggregate.py       --config configs/dev.yaml   # tables (md / csv / tex)
python scripts/make_figures.py    --config configs/dev.yaml   # figures (pdf / png)
```

`bash scripts/run_all.sh configs/dev.yaml` runs that whole sequence. Results
land in `runs/<name>/`. Any config value can be overridden on the command
line, e.g. `nested.sync.bits=4` or `stream.kind=temporal`.

## Configs

| File | Backbone | Purpose |
| --- | --- | --- |
| `configs/smoke.yaml` | tiny random model, toy text | offline pipeline check; numbers are meaningless |
| `configs/dev.yaml` | SmolLM2-135M | first real run, debugging, sweeps |
| `configs/main.yaml` | SmolLM2-360M | headline results |
| `configs/scale.yaml` | Qwen2.5-0.5B | second backbone family |

## Layout

```
dnsc/            library
  codec.py       the rate-nested LLM codec (transmitter, receiver, parameter groups)
  channel.py     AWGN / Rayleigh block fading, shell-wise power constraint
  nested.py      multi-timescale optimiser, update quantisation, sync ledger
  online.py      prequential drift-stream runner and all baselines
  refine.py      uncertainty-triggered incremental refinement + calibration
  classical.py   ideal separation-based references (derived)
  linkbudget.py  net semantic rate and adaptation crossover (derived)
  metrics.py     BLEU-1..4, sacreBLEU, sentence similarity, ECE
  stats.py       paired bootstrap / permutation, Holm, equivalence test
  data/          corpus loaders and drift-stream construction
scripts/         one entry point per experiment stage
configs/         experiment definitions
tests/           unit and integration tests
docs/            method, datasets, roadmap
```

## What has and has not been verified

Verified in the build environment (Linux, CPU, PyTorch 2.14, Transformers 5.19):

- the full test suite, including every script run from the shell;
- on the tiny model: the codec overfits a small set to 100 % token accuracy,
  and the cached greedy decoder agrees token-for-token with a full forward pass;
- the code path used for real backbones (`AutoModelForCausalLM` / `AutoTokenizer`
  by name, bfloat16 checkpoint, frozen, adapters only on disk), using a small
  locally saved model;
- the AG News and TeleQnA loaders, against the real sources.

**Not** verified there, because the build environment could not reach the
Hugging Face Hub and has no Apple GPU:

- downloading and training an actual pretrained backbone (SmolLM2 / Qwen2.5),
  and therefore how well a frozen small LLM reconstructs real sentences;
- execution on Apple MPS;
- the Europarl and HuffPost loaders and the sentence-similarity model.

`scripts/check_env.py` exercises exactly these on your machine and reports
PASS/FAIL per item, so run it first.

## Reproducibility rules this repository follows

- Every reported number is produced by code and written to a JSON file together
  with the full config, library versions and git revision.
- Tables are generated from those files by `scripts/aggregate.py`; none is typed.
- Splits are by document; probes and test sentences are never trained on;
  stream metrics are prequential (scored before the update).
- Derived quantities (link budget, separation references) are labelled
  "derived" and kept apart from measurements.
- Negative results are reported as they fall.

## Licence

Code: MIT. Datasets keep their own licences, see `docs/DATASETS.md`.
