# Doubly Nested Semantic Communication (DN-SC)

Research code and results for a rate- and time-nested LLM semantic link for
drift-resilient 6G networks, prepared for the *International Journal of
Communication Systems* special issue "Next-Generation 6G Mobile Networks and
Services: LLM-Enabled Semantic Communications" (submission deadline
31 March 2027).

**Status: v0.2.** The method, the full experiment pipeline and a first
complete set of results with a pretrained LLM (GPT-2, 124M) are in this
repository: see [`results/gpt2/RESULTS.md`](results/gpt2/RESULTS.md).
Version 0.1 (a sentence-level soft-prompt autoencoder) could not be trained on
a CPU; that negative result is recorded in
[`docs/METHOD.md`](docs/METHOD.md), section 8.

## The idea in five lines

- **Tokens over the air, an LLM as the prior.** Every token is sent as a few
  real symbols from a learned codebook. The receiver combines what it hears
  with what a frozen LLM expects next, and decides.
- **Nested in rate.** The symbols of a token are ordered: *any prefix* is a
  valid transmission, with exact power control. The payload adapts to the link.
- **Nested in time.** The receiver's prior has three memories: the
  conversation so far as context (every message, free), a logit bias
  (minutes, kilobits) and LoRA adapters kept in slots, one per knowledge
  domain (hours, megabits). The slower ones must be shipped over the air, and
  every shipped bit is counted.
- **Net semantic rate.** Payload plus amortised synchronisation symbols per
  message — the quantity that decides whether adapting pays for itself, and
  above which traffic volume.
- **Selective refinement.** The receiver asks for more symbols only for the
  tokens it is unsure about: token-level incremental redundancy driven by the
  LLM's calibrated uncertainty.

Details: [`docs/METHOD.md`](docs/METHOD.md) · datasets:
[`docs/DATASETS.md`](docs/DATASETS.md) · plan:
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

### Reproducing the GPT-2 results (`configs/gpt2.yaml`)

Everything this configuration needs comes from GitHub, so it also runs where
Hugging Face is unreachable.

```
python -m pip install onnx
python scripts/fetch_gpt2.py --out models/gpt2
python scripts/run_pipeline.py --config configs/gpt2.yaml --stage base
python scripts/run_pipeline.py --config configs/gpt2.yaml --stage pilot
python scripts/run_pipeline.py --config configs/gpt2.yaml --stage streams --seeds 0,1,2
python scripts/run_pipeline.py --config configs/gpt2.yaml --stage report
```

On two CPU cores the codebook trains in about 40 minutes and an adaptive
stream run takes about 30 minutes; `scripts/benchmark_speed.py` prints the
figures for your machine. Where Hugging Face is reachable, skip the first two
lines and add `backbone.name=openai-community/gpt2` to every command.

### A second backbone on a CPU-only laptop (`configs/laptop.yaml`)

Same method with SmolLM2-135M and the Hugging Face corpora:

```
python scripts/run_pipeline.py --config configs/laptop.yaml --stage base
python scripts/run_pipeline.py --config configs/laptop.yaml --stage pilot
python scripts/run_pipeline.py --config configs/laptop.yaml --stage streams --seeds 0,1,2
python scripts/run_pipeline.py --config configs/laptop.yaml --stage report
```

- `base` = environment check, data, speed benchmark, codebook pretraining,
  rate and refinement evaluation, and the memory-headroom diagnostic.
- `pilot` = one seed of the five key methods on the hierarchical stream, then
  the tables: the cheapest way to see the whole picture.
- `streams` = both drift streams for all methods and seeds. Finished runs are
  skipped on re-start, so it can be run a few hours at a time.
- `report` = tables and figures from whatever results exist.
- Add `threads=4` or `threads=8` at the end of a command to set the CPU
  threads. Keep the laptop on mains power and stop it sleeping while a stage
  runs (Windows: Settings → System → Power → "When plugged in, put my device
  to sleep after" → Never).

Each stage can also be run script by script (`scripts/pretrain.py`,
`scripts/run_stream.py`, ...); every script accepts `--config` and any number
of `section.key=value` overrides, e.g. `nested.sync.bits=4`.
Results land in the config's `out_dir`.

## Configs

| File | Language prior | Purpose |
| --- | --- | --- |
| `configs/smoke.yaml` | tiny model trained on toy text | pipeline check; numbers are meaningless |
| `configs/rehearsal.yaml` | small model trained from scratch on AG News | whole pipeline on real text with no pretrained model; not a result |
| `configs/gpt2.yaml` | GPT-2 (124M) | **the results in `results/gpt2`**; all inputs from GitHub |
| `configs/laptop.yaml` | SmolLM2-135M | second backbone, sized for a CPU-only laptop |
| `configs/main.yaml` | SmolLM2-360M | larger backbone (a GPU is advisable) |
| `configs/scale.yaml` | Qwen2.5-0.5B | another backbone family (a GPU is advisable) |

## Layout

```
dnsc/            library
  codec.py       semantic codebook, block power control, Bayesian token decoder, memories
  channel.py     AWGN / Rayleigh block fading, nested power constraint
  nested.py      working copy vs receiver copy, update quantisation, sync ledger
  online.py      prequential drift-stream runner, slot selection and all baselines
  refine.py      selective refinement, calibration, controls
  headroom.py    diagnostic: what each memory could buy at best
  classical.py   ideal separation-based references (derived)
  linkbudget.py  net semantic rate and adaptation crossover (derived)
  metrics.py     BLEU-1..4, sacreBLEU, sentence similarity, ECE
  stats.py       paired bootstrap / permutation, block bootstrap, Holm, equivalence test
  data/          corpus loaders and document-coherent drift streams
scripts/         one entry point per experiment stage
configs/         experiment definitions
results/         exported tables, figures and compact result files
tests/           unit and integration tests
docs/            method, datasets, roadmap
```

## What has and has not been verified

Verified by the test suite (Linux, CPU, PyTorch 2.14, Transformers 5.19):

- exactness properties: unit transmit power for every prefix of every
  message; the likelihood against a brute-force computation; error-free
  decoding on a clean channel without any prior; the cached greedy decoder
  against a full forward pass, with and without context; a padded batch
  against single messages; receiver parameters changing only when a level is
  shipped, with a bit-exact ledger;
- every script run from the shell, and the code path used for real backbones
  (Llama-style and GPT-2-style layers).

Measured with a pretrained LLM: everything in `results/gpt2`, produced by the
commands above on two CPU cores.

**Not done yet:**

- a second backbone (`configs/laptop.yaml` is ready for it);
- sentence-embedding similarity (needs a model from Hugging Face; the decoded
  texts are stored by `eval_rate.py`, so it can be added without re-running);
- execution on CUDA or Apple MPS.

## Reproducibility rules this repository follows

- Every reported number is produced by code and written to a JSON file together
  with the full config, library versions and git revision.
- Tables are generated from those files by `scripts/aggregate.py`; none is typed.
- Splits are by document; probes and test sentences are never trained on;
  stream metrics are prequential (the receiver is scored before any update,
  with the copy of the parameters it actually holds).
- Derived quantities (link budget, separation references) and diagnostics with
  oracle inputs (headroom, `llm_ctx`) are labelled as such and kept apart from
  results.
- Negative results are reported as they fall.

## Licence

Code: MIT. Datasets and the GPT-2 weights keep their own licences, see
`docs/DATASETS.md`.
