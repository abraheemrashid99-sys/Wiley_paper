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

## Quick start

Python 3.10 or newer. The commands are the same in Windows PowerShell, macOS
and Linux (on macOS/Linux use `python3` if `python` is not found).

```
git clone https://github.com/abraheemrashid99-sys/Wiley_paper.git
cd Wiley_paper
python -m pip install -e ".[dev]"

python -m pytest
python scripts/run_pipeline.py --config configs/smoke.yaml --stage smoke
```

`pytest` runs the unit and integration tests offline; the `smoke` stage runs
the whole pipeline on a tiny offline model. Both take a few minutes.

### On a CPU-only laptop (`configs/laptop.yaml`)

```
python scripts/benchmark_speed.py --config configs/laptop.yaml
python scripts/run_pipeline.py --config configs/laptop.yaml --stage base
python scripts/run_pipeline.py --config configs/laptop.yaml --stage streams --seeds 0,1,2
python scripts/run_pipeline.py --config configs/laptop.yaml --stage report
```

- `benchmark_speed.py` prints how long pretraining and each stream run will
  take on this machine. Try `threads=4` and `threads=8` after the config name
  and keep the faster one by adding it to every later command.
- `base` = environment check, data, pretraining, rate and refinement
  evaluation. Pretraining is the long step and is **resumable**: if the laptop
  sleeps or you press Ctrl+C, run the same command again.
- `streams` = both drift streams for all methods. Finished runs are skipped on
  re-start, so it can be run a few hours at a time.
- Keep the laptop on mains power and stop it sleeping while a stage runs
  (Windows: Settings → System → Power → "When plugged in, put my device to
  sleep after" → Never).

Each stage can also be run script by script (`scripts/pretrain.py`,
`scripts/run_stream.py`, ...); every script accepts `--config` and any number
of `section.key=value` overrides, e.g. `nested.sync.bits=4`.
Results land in `runs/<name>/`.

### With a GPU

`configs/dev.yaml`, `configs/main.yaml` and `configs/scale.yaml` use larger
backbones and longer streams and need a GPU: an NVIDIA card (CUDA) or Apple
Silicon (MPS) is picked up automatically. A free Kaggle or Colab T4 runs the
same commands after `git clone` and `pip install -e .`; copy `runs/` out before
the session ends. The CUDA and MPS paths have not been exercised yet.

## Configs

| File | Backbone | Purpose |
| --- | --- | --- |
| `configs/smoke.yaml` | tiny offline model, toy text | pipeline check; numbers are meaningless |
| `configs/laptop.yaml` | SmolLM2-135M | **CPU-only laptop: the full study at reduced compute** |
| `configs/dev.yaml` | SmolLM2-135M | GPU rehearsal, sweeps |
| `configs/main.yaml` | SmolLM2-360M | GPU headline results |
| `configs/scale.yaml` | Qwen2.5-0.5B | GPU, second backbone family |

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

- the full test suite, including every script run from the shell, bit-for-bit
  resumption of pretraining, and the code path used for real backbones;
- on the tiny model: the codec memorises a small set to 100 % token accuracy,
  and the cached greedy decoder agrees token-for-token with a full forward pass;
- the AG News and TeleQnA loaders, against the real sources.

Verified on the target laptop (Windows 11, Python 3.14, CPU), by
`scripts/check_env.py`:

- SmolLM2-135M downloads, and a forward/backward/generate step runs;
- all four corpora build (Europarl, AG News, TeleQnA, HuffPost 2012–2022);
- the sentence-similarity model loads.

**Not yet verified anywhere:**

- how well a frozen small LLM reconstructs real sentences after pretraining —
  the first gate in the roadmap;
- execution on CUDA or Apple MPS.

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
