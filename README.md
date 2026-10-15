# Doubly Nested Semantic Communication (DN-SC)

Research code and results for a rate- and time-nested LLM semantic link for
drift-resilient 6G networks, prepared for the *International Journal of
Communication Systems* special issue "Next-Generation 6G Mobile Networks and
Services: LLM-Enabled Semantic Communications" (submission deadline
31 March 2027).

**Status: v0.4.** The method, the full experiment pipeline and a first
complete set of results with a pretrained LLM (GPT-2, 124M) are in this
repository: see [`results/gpt2/RESULTS.md`](results/gpt2/RESULTS.md). That
result set (version 0.2) was reviewed strictly
([`docs/REVIEW.md`](docs/REVIEW.md) lists every finding and what was done
about it) and reproduced independently on a GPU (all 37 runs, REPRODUCED).
Version 0.3 added the measurements the review asked for and the confirmatory
protocol ([`docs/METHOD.md`](docs/METHOD.md), section 9); they show that,
decoded for real and with its return-link signalling charged, the full
system saves about a quarter of the channel symbols of a frozen fixed-rate
receiver, not half. Version 0.4 replaces the refinement decoder by one chosen
on validation text only (beam search, a first round of 6 symbols per token).
**The confirmatory runs have not been made yet**, so the numbers in
`results/gpt2` are to be read as a pilot. Version 0.1 (a
sentence-level soft-prompt autoencoder) could not be trained on a CPU; that
negative result is recorded in `docs/METHOD.md`, section 8.

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
  LLM's calibrated uncertainty. The requests travel on the return link, and
  that signalling is counted too.

Details: [`docs/METHOD.md`](docs/METHOD.md) · review:
[`docs/REVIEW.md`](docs/REVIEW.md) · datasets:
[`docs/DATASETS.md`](docs/DATASETS.md) · plan:
[`docs/ROADMAP.md`](docs/ROADMAP.md) · changes:
[`CHANGELOG.md`](CHANGELOG.md).

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

**In VS Code** the same steps, and the reproduction below, are ready-made
tasks (Terminal → Run Task…): see [`RUN_IN_VSCODE.md`](RUN_IN_VSCODE.md).

### Reproducing and extending the GPT-2 study (`configs/gpt2.yaml`)

Everything this configuration needs comes from GitHub, so it also runs where
Hugging Face is unreachable. One command runs a whole study from scratch and
compares it with the published numbers:

```
python -m pip install -e ".[dev,gpt2]"
python scripts/fetch_gpt2.py --out models/gpt2
python scripts/reproduce.py --config configs/gpt2.yaml --out runs/gpt2_published \
    --export results_repro/gpt2_published --reference results/gpt2 --profile published
```

| `--profile` | Runs | Purpose |
| --- | --- | --- |
| `published` | the 37 stream runs behind `results/gpt2` and every evaluation | check the published numbers |
| `core` | main methods and the time-nesting ablation on six seeds on both streams; other ablations and the Rayleigh stream on three | the confirmatory protocol of `docs/METHOD.md`, section 9 |
| `paper` | `core` plus the sensitivity studies (shipping periods, quantiser, drift speed, mean SNR, context length, adapter rank, three domains against two and three slots) | everything the paper reports |

Each profile contains the one before it under the same file names. The
command can be repeated after an interruption (finished runs are skipped) and
ends with `COMPARISON.md` in the export folder: provenance, the published and
the reproduced value of every number, a tolerance check of the headline
numbers, a check of each conclusion, and what only the new run measures (the
signalling of refinement, the free-running audit). `--dry-run` prints the
plan.

On two CPU cores the codebook trains in about 40 minutes, an adaptive stream
run takes about 30 minutes and one without adaptation about 10; the second
step of the command measures the figures for your machine. Where Hugging
Face is reachable, `backbone.name=openai-community/gpt2` at the end of the
command replaces the second line. The stages of the study can also be run
one by one with `scripts/run_pipeline.py --stage base|pilot|streams|report`.

### Checking the reproduction on a GPU (Google Colab)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abraheemrashid99-sys/Wiley_paper/blob/main/notebooks/reproduce_gpt2_colab.ipynb)

`notebooks/reproduce_gpt2_colab.ipynb` runs the same command on a T4 GPU:
it installs the repository, runs the test suite and the complete protocol on
a tiny model as a GPU self-test, then the chosen profile with GPT-2, and
displays `COMPARISON.md`. A CPU and a GPU do not produce bit-identical
numbers, so the comparison against a tolerance and against the conclusions
is the meaningful test; on the same machine, with the same library versions,
the runs are repeatable bit for bit (`requirements-lock.txt` lists the
versions behind `results/gpt2`). Run with `PROFILE = "published"` on a Colab
T4, it reproduced all 37 published runs (`results/gpt2_v0.3_colab_t4`); its
default is now `"core"`, the confirmatory study.

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
scripts/         one entry point per experiment stage; reproduce.py runs them all,
                 compare_results.py checks a run against published numbers,
                 make_release.py / verify_package.py build and check the archive
notebooks/       Colab notebook that reproduces results/gpt2 on a GPU
.vscode/         ready-made tasks for VS Code (see RUN_IN_VSCODE.md)
configs/         experiment definitions
results/         exported tables, figures and compact result files
tests/           unit and integration tests
tools/           device audit (virtual GPU on a CPU-only machine)
docs/            method, review, analysis notes, datasets, roadmap
```

## What has and has not been verified

Verified by the test suite (Linux, CPU, PyTorch 2.14, Transformers 5.19;
86 tests):

- exactness properties: unit transmit power for every prefix of every
  message; the likelihood against a brute-force computation; error-free
  decoding on a clean channel without any prior; the cached greedy decoder
  against a full forward pass, with and without context; a padded batch
  against single messages; receiver parameters changing only when a level is
  shipped, with a bit-exact ledger;
- the accounting of refinement signalling against hand-computed cases; the
  free-running refinement decoder against the emulation where the two must
  coincide (one-token messages) and at its limits; beam search against greedy
  decoding at width one and against exhaustive search at full width; that the
  audit moves no other number; the bootstrap of the real-decoding endpoint;
- every script run from the shell, the reproduction plans of all profiles,
  the comparison report on a matching and on a tampered folder, and the code
  path used for real backbones (Llama-style and GPT-2-style layers).

Measured with a pretrained LLM: everything in `results/gpt2` (version 0.2,
two CPU cores). For version 0.3 the published codebook was regenerated
(training log identical line by line) and 6 of the 37 published
stream runs were repeated with the corrected code: all are identical to the
published per-step logs in every measured value. `results/gpt2/RESULTS.md`,
section 11, reports this and the first values of the new measurements;
section 12 the reproduction of all 37 runs on a Colab T4 and the new
measurements on every seed. The decoder selection of version 0.4 is in
`results/decoder_selection`.

**Not done yet:**

- the confirmatory runs (`--profile core`, with the decoder chosen in
  version 0.4) and the sensitivity studies (`--profile paper`);
- a second backbone (`configs/laptop.yaml` is ready for it) and
  sentence-embedding similarity (`scripts/add_similarity.py`); both need
  Hugging Face;
- a trained end-to-end baseline, and decoding at low SNR;
- execution on macOS. The published profile has run on a Colab T4 GPU and
  the test suite on Windows 11; Apple MPS is untested.

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
- A result that is emulated is audited against the real thing on a subset,
  and every cost a mechanism causes is charged to it, also on the return link.
- "No difference" is claimed only through an equivalence test with a margin
  fixed beforehand; a difference only if it also holds at the seed level.
- A release is one commit: the archive is built from it, carries its
  revision and a checksum of every file, and results record the revision.

## Licence

Code: MIT. Datasets and the GPT-2 weights keep their own licences, see
`docs/DATASETS.md`.
