# Roadmap: 15 days of experiments, then writing

Target: *International Journal of Communication Systems* special issue on
LLM-enabled semantic communications, deadline **31 March 2027**.
Hardware: MacBook Air M4, 16 GB, no fan — sustained runs will throttle, so
long jobs are scheduled overnight and the laptop should be on mains power,
lid open, on a hard surface.

A plan cannot promise results. It can make sure that by day 15 every claim in
`docs/METHOD.md` has been tested properly, whichever way it fell. Three gates
decide early whether the direction holds.

## Compute budget

Speed on your machine is unknown until `scripts/benchmark_speed.py` is run; it
prints hours per pretraining and per stream run. Plan with those numbers:

```
total hours ≈ pretraining + (adaptive methods × seeds × streams) × hours per stream run
```

`frozen` is cheap (no backward pass). If the main config is too slow, the
order of sacrifice is: fewer seeds for ablations (keep 5 for the headline
pair), shorter streams, then SmolLM2-135M as the headline backbone with 360M
as the scale check.

## Days 1–15

| Day | Work | Output | Gate |
| --- | --- | --- | --- |
| 1 | Clone, install, `pytest`, `run_smoke.sh`, `check_env.py`, `prepare_data.py`, `benchmark_speed.py` for dev and main. Start dev pretraining overnight. | Working environment, data cache, measured speed | **Gate 0:** all checks PASS |
| 2 | `eval_rate.py` on dev. Inspect reconstructions by eye. Start main pretraining overnight. | Rate–fidelity table, dev | **Gate 1:** base codec reconstructs (see below) |
| 3 | `eval_rate.py` on main. Train like-for-like fixed-rate codecs on dev (`codec.granularities=[64] stream.m_op=64`, and `[256]`). | Claim 1 evidence | Equivalence margin fixed before looking |
| 4 | Dev rehearsal of the `hier` stream: all 8 methods × 3 seeds. | First stream table | **Gate 2:** drift hurts `frozen`, adaptation recovers |
| 5–6 | Main `hier` stream: headline methods × 5 seeds, ablations × 3. | Headline table 1 | — |
| 7 | Main `temporal` stream (HuffPost): headline methods × 5 seeds. | Headline table 2 | **Gate 3:** does the effect survive natural drift? |
| 8 | Sensitivity: periods, sync bits (2/4/8) and top-k (1.0/0.25/0.1), `tau_slow`, `tau_med`, `kappa`. | Crossover and drift-structure curves (claims 3, 4) | — |
| 9 | `eval_refine.py` on base and adapted codecs, AWGN and Rayleigh. | Claim 5 evidence | Beats the random control? |
| 10 | Second backbone (`configs/scale.yaml`): pretrain + headline pair × 3 seeds. | Backbone-independence | — |
| 11 | Robustness: train-AWGN/test-Rayleigh, SNR mismatch, other `m_op`. | Robustness table | — |
| 12 | Efficiency: measured latency and memory; energy via `powermetrics`; derived link budget. | Cost tables | — |
| 13 | Fill gaps: failed runs, missing seeds. Statistics pass (Holm, equivalence). | Complete result set | — |
| 14 | `aggregate.py` + `make_figures.py` for everything. Literature sweep to check the "first" claims. | All tables and figures | — |
| 15 | Freeze: tag the commit, archive `runs/`, re-run one experiment from a clean clone and confirm the numbers match. | Frozen, reproducible results | — |

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

- DeepSC-style Transformer JSCC baseline trained from scratch (day 3).
- Sweep driver for the day-8 sensitivity grid.
- Energy measurement wrapper around `powermetrics` (day 12).
- Channel/SNR mismatch options for day 11.

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
