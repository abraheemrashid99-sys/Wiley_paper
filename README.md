# Doubly Nested Semantic Communication (DN-SC)

Research code for a rate- and time-nested LLM semantic link for drift-resilient
6G networks, prepared for the *International Journal of Communication Systems*
special issue "Next-Generation 6G Mobile Networks and Services: LLM-Enabled
Semantic Communications" (submission deadline 31 March 2027).

**Status: v0.2 — token-level redesign for CPU-only hardware. The pipeline is
complete and tested; no result of the paper has been produced yet.** The first
run with a pretrained LLM happens on your machine (see "What has and has not
been verified"). Version 0.1 (a sentence-level soft-prompt autoencoder) could
not be trained on a CPU; that negative result is recorded in
[`docs/METHOD.md`](docs/METHOD.md), section 8.

## The idea in five lines

- **Tokens over the air, an LLM as the prior.** Every token is sent as a few
  real symbols from a learned codebook. The receiver combines what it hears
  with what a frozen LLM expects next, and decides.
- **Nested in rate.** The symbols of a token are ordered: *any prefix* is a
  valid transmission, with exact power control. The payload adapts to the link.
- **Nested in time.** The receiver's prior has three memories — a token cache
  (every message, free), a logit bias (minutes, kilobits) and a LoRA adapter
  (hours, megabits). The slower ones must be shipped over the air, and every
  shipped bit is counted.
- **Net semantic rate.** Payload plus amortised synchronisation symbols per
  message — the quantity that decides whether adapting pays for itself, and
  above which traffic volume.
- **Selective refinement.** The receiver asks for more symbols only for the
  tokens it is unsure about: token-level incremental redundancy driven by the
  LLM's calibrated uncertainty.

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
python scripts/run_pipeline.py --config configs/laptop.yaml --stage base
python scripts/run_pipeline.py --config configs/laptop.yaml --stage pilot
python scripts/run_pipeline.py --config configs/laptop.yaml --stage streams --seeds 0,1,2
python scripts/run_pipeline.py --config configs/laptop.yaml --stage report
```

- `base` = environment check, data, speed benchmark, codebook pretraining,
  rate and refinement evaluation, and the adaptation-headroom diagnostic.
  The benchmark prints what every later stage will cost on this machine.
- `pilot` = one seed of the five key methods on the hierarchical stream, then
  the tables. It is the cheapest way to see whether adaptation pays; look at it
  before starting the long runs.
- `streams` = both drift streams for all methods and seeds. Finished runs are
  skipped on re-start, so it can be run a few hours at a time.
- `report` = tables and figures from whatever results exist.
- Add `threads=4` or `threads=8` at the end of a command to set the CPU
  threads; `scripts/benchmark_speed.py` shows which is faster.
- Keep the laptop on mains power and stop it sleeping while a stage runs
  (Windows: Settings → System → Power → "When plugged in, put my device to
  sleep after" → Never).

Each stage can also be run script by script (`scripts/pretrain.py`,
`scripts/run_stream.py`, ...); every script accepts `--config` and any number
of `section.key=value` overrides, e.g. `nested.sync.bits=4`.
Results land in the config's `out_dir` (`runs/laptop_v2/` for the laptop).

## Configs

| File | Language prior | Purpose |
| --- | --- | --- |
| `configs/smoke.yaml` | tiny model trained on toy text | pipeline check; numbers are meaningless |
| `configs/rehearsal.yaml` | small model trained from scratch on AG News | whole method on real text with no pretrained model and no Hugging Face access; not a result |
| `configs/laptop.yaml` | SmolLM2-135M | **the study, on a CPU-only laptop** |
| `configs/main.yaml` | SmolLM2-360M | second, larger backbone (CPU for a few runs, or a free GPU) |
| `configs/scale.yaml` | Qwen2.5-0.5B | second backbone family |

## Layout

```
dnsc/            library
  codec.py       semantic codebook, block power control, Bayesian token decoder, memories
  channel.py     AWGN / Rayleigh block fading, nested power constraint
  nested.py      working copy vs receiver copy, update quantisation, sync ledger
  online.py      prequential drift-stream runner and all baselines
  refine.py      selective refinement, calibration, controls
  headroom.py    diagnostic: what each memory could buy at best
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

- the full test suite, including every script run from the shell and the code
  path used for real backbones;
- exactness properties: unit transmit power for every prefix of every
  message; the likelihood against a brute-force computation; error-free
  decoding on a clean channel without any prior; the cached greedy decoder
  against a full forward pass; receiver parameters changing only when a level
  is shipped, with a bit-exact ledger;
- the whole pipeline on **real text** with `configs/rehearsal.yaml` (AG News
  and TeleQnA; a small language model trained from scratch stands in for the
  LLM). There the codec behaves as designed: the language prior buys rate,
  fidelity degrades gracefully where ideal separation falls off a cliff, the
  token confidence is calibrated, and selective refinement saves symbols at
  equal accuracy. See `docs/ROADMAP.md` for the numbers.

**Not verified, and the first thing to look at on the laptop:**

- everything that depends on a real pretrained LLM: how much rate SmolLM2
  buys, and above all **how much there is to gain by adapting it**. In the
  rehearsal the from-scratch model had little adaptation headroom, so the
  time-nesting claim is untested. `scripts/headroom.py` (part of the `base`
  stage) measures this directly in a few minutes;
- execution on CUDA or Apple MPS.

## Reproducibility rules this repository follows

- Every reported number is produced by code and written to a JSON file together
  with the full config, library versions and git revision.
- Tables are generated from those files by `scripts/aggregate.py`; none is typed.
- Splits are by document; probes and test sentences are never trained on;
  stream metrics are prequential (the receiver is scored before any update,
  with the copy of the parameters it actually holds).
- Derived quantities (link budget, separation references) and diagnostics with
  oracle inputs (headroom) are labelled as such and kept apart from results.
- Negative results are reported as they fall.

## Licence

Code: MIT. Datasets keep their own licences, see `docs/DATASETS.md`.
