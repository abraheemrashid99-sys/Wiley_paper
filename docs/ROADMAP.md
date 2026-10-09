# Roadmap: 15 days of experiments, then writing

Target: *International Journal of Communication Systems* special issue on
LLM-enabled semantic communications, deadline **31 March 2027**.
Hardware: ASUS ZenBook, Ryzen 7 3700U (4 cores), 16 GB, integrated graphics —
**CPU only**. Everything below is sized for `configs/laptop.yaml`. Keep the
laptop on mains power with sleep disabled while a stage runs.

A plan cannot promise results. It can make sure that by day 15 every claim in
`docs/METHOD.md` has been tested properly, whichever way it fell. Three gates
decide early whether the direction holds.

## What the CPU changes

The first benchmark on this laptop measured 0.07 training steps per second
for SmolLM2-135M at batch size 16 — about 16 hours for a 4 000-step
pretraining and 4 hours per stream run, which does not fit in 15 days.
`configs/laptop.yaml` therefore cuts the work per step roughly three- to
four-fold (batch size 8, one prefix length per step, sentences of 4–20 words)
and shortens the streams to 600 steps. Expected order of magnitude, to be
replaced by what `scripts/benchmark_speed.py` prints:

| Stage | Rough cost on this laptop |
| --- | --- |
| Pretraining, 4 000 steps (resumable) | 5–9 h, one or two nights |
| One adaptive stream run (600 steps) | about 1 h |
| `frozen` stream run | about 20 min |
| 7 methods × 3 seeds × 2 streams | 35–40 h, spread over a week of nights |
| Rate and refinement evaluation | 1–2 h |

What this costs scientifically, stated plainly:

- The headline backbone is SmolLM2-135M. A reviewer may ask for a larger
  model; `configs/main.yaml` and `configs/scale.yaml` are ready for that and
  take a few hours on a free Kaggle or Colab GPU.
- Three seeds for every method, five only for the headline pair
  (`nested` vs `single_rate`).
- Sensitivity sweeps must be chosen sparingly (day 8).

If the benchmark shows the budget still does not fit, the order of sacrifice
is: fewer ablation seeds, `tau_slow=100`, then moving the long runs to a free
GPU.

## Days 1–15

| Day | Work | Output | Gate |
| --- | --- | --- | --- |
| 1 | `git pull`, install, `pytest`, smoke stage, `benchmark_speed.py` (try `threads=4` and `threads=8`). Start `--stage base` overnight. | Measured speed; pretraining under way | **Gate 0:** tests and smoke pass |
| 2 | Pretraining finishes (resume if interrupted); rate and refinement evaluation follow automatically. Read the reconstructions. | Rate–fidelity table | **Gate 1:** base codec reconstructs (see below) |
| 3 | If Gate 1 fails, apply the remedies and retrain. Otherwise train one like-for-like fixed-rate codec overnight (`codec.granularities=[32] out_dir=runs/laptop_fixed32`). | Claim 1 evidence | Equivalence margin fixed before looking |
| 4 | `hier` stream, seed 0, all methods (overnight). | First stream table | **Gate 2:** drift hurts `frozen`, adaptation recovers |
| 5–6 | `hier` stream, seeds 1–2; seeds 3–4 for `nested` and `single_rate`. | Headline table 1 | — |
| 7–8 | `temporal` stream (HuffPost), same seeds. | Headline table 2 | **Gate 3:** does the effect survive natural drift? |
| 9 | Sensitivity, one seed each: sync bits 4 and top-k 0.25; `tau_slow` 75 and 300; periods (5, 25) and (20, 100). | Crossover and drift-structure evidence (claims 3, 4) | — |
| 10 | Refinement on Rayleigh and on an adapted codec. | Claim 5 evidence | Beats the random control? |
| 11 | Robustness: train-AWGN/test-Rayleigh, other `m_op`. | Robustness table | — |
| 12 | Measured latency and memory; derived link budget. Optional: `configs/main.yaml` on a free GPU. | Cost tables | — |
| 13 | Fill gaps: failed runs, missing seeds. Statistics pass (Holm, equivalence). | Complete result set | — |
| 14 | `--stage report` for everything. Literature sweep to check the "first" claims. | All tables and figures | — |
| 15 | Freeze: tag the commit, archive `runs/`, re-run one stream run from a clean clone and confirm the numbers match. | Frozen, reproducible results | — |

### Gate 1 — does the base codec work?

At the longest prefix and 20 dB the reconstructions should be mostly correct
sentences, and fidelity should rise with both `m` and SNR. If it does not, in
order: train longer (`pretrain.steps`), raise capacity (`codec.n_slots`,
`codec.n_prompts`, LoRA rank, `rx_last_n=0`), then narrow the training SNR
range. If a frozen small LLM still cannot reconstruct, that is a finding about
the approach and we rescope before going further.

### Gate 2 — is there anything to adapt to?

`frozen` must be clearly worse on the stream domains than on its training
domain, and `full_online` must recover most of that gap. If `frozen` is fine,
the drift is too weak to study; if `full_online` cannot recover, the adapters
are too small. Either changes the design, not the reporting.

### Gate 3 — the honest test

If `nested` beats `single_rate` on `hier` but not on `temporal`, the paper says
so and is framed around *when* time nesting pays. If it wins on neither, the
paper becomes the net-semantic-rate and crossover analysis with rate nesting
and calibrated refinement — still a complete contribution for this call.

## Code still to add (not in v0.1)

- DeepSC-style Transformer JSCC baseline trained from scratch (day 3; small enough for the CPU).
- Sweep driver for the day-9 sensitivity grid.
- Channel/SNR mismatch options for day 11.
- Energy measurement (needs a platform-specific power reader).

## Writing (from day 16)

Suggested structure, mirroring the standards of your earlier papers:

1. Introduction — multi-timescale semantic drift; the receiver cannot adapt alone.
2. Related work with a synthesis table of evidential controls.
3. System model and the net semantic rate.
4. Method: rate nesting, time nesting, calibrated refinement.
5. Analysis: finite-time bound with quantised slow updates.
6. Experimental setup — everything fixed before results.
7. Results in the order of the claims.
8. Discussion: what the evidence supports and what it does not; design rules.
9. Limitations; conclusion.

Topics of the call covered: LLM-enabled semantic communication frameworks;
semantic compression and task-oriented strategies; LLMs at the edge;
distributed LLMs for communication systems; security/trust and explainability
(calibrated confidence); energy-efficient solutions; experimental platforms.

Before submission: read the journal's author guidelines for length, template
and the policy on AI-assisted writing, and disclose AI assistance as you did
in the ECG paper.
