# Roadmap

Target: *International Journal of Communication Systems* special issue on
LLM-enabled semantic communications, deadline **31 March 2027**.

## Where the project stands

| Step | State |
| --- | --- |
| Method fixed and written down (`docs/METHOD.md`) | done |
| Pipeline: codebook, receiver, streams, baselines, statistics, tables, figures | done, tested |
| First complete result set with a pretrained LLM (GPT-2, 124M) | done: `results/gpt2/RESULTS.md` |
| Second backbone (SmolLM2-135M, `configs/laptop.yaml`) | to run on the laptop |
| Sentence-embedding similarity as an extra fidelity metric | to add where Hugging Face is reachable |
| Systematic literature sweep for the "first" claims | to do before writing |
| Manuscript | to write |

## What the first result set already decides

Read `results/gpt2/RESULTS.md` for the numbers. In one paragraph: the three
free or cheap mechanisms — selective refinement, the LLM prior and the
context memory — carry most of the gain; shipped adaptation adds a smaller,
statistically clear gain, reaches the per-message upper reference at a few
percent of its synchronisation cost, and pays for itself only above a traffic
volume that the crossover analysis makes explicit. The paper should be framed
around that economy, not around adaptation alone.

## Remaining experiments

### A. Second backbone on the laptop (about two days of machine time)

```
git pull
python -m pip install -e ".[dev]"
python -m pytest
python scripts/run_pipeline.py --config configs/laptop.yaml --stage base
python scripts/run_pipeline.py --config configs/laptop.yaml --stage pilot
python scripts/run_pipeline.py --config configs/laptop.yaml --stage streams --seeds 0,1,2
python scripts/run_pipeline.py --config configs/laptop.yaml --stage report
```

`base` prints a speed benchmark first; if an adaptive stream run is estimated
above one hour, add `stream.batch_size=6` to every command (the GPT-2 study
used 6). The old `runs/laptop` directory belongs to version 0.1 and can be
deleted. Gates:

1. **After `base`** — the rate table must show the same ordering as for
   GPT-2 (`llm_ctx` ≥ `llm` > `unigram` ≥ `none` at low rate), and the
   headroom table must show a gain from context and from adaptation on
   TeleQnA. If it does not, stop and compare with `results/gpt2`.
2. **After `pilot`** — `context_only` must beat `frozen`, and `nested` must
   not be worse than `context_only`.

### B. Checks a reviewer will ask for

| Check | How | Cost |
| --- | --- | --- |
| Temporal stream on the official HuffPost release | `stream.temporal_corpus=huffpost` on the laptop config | part of A |
| Sentence similarity (SBERT) on the rate results | from the stored decoded texts, where Hugging Face is reachable | minutes |
| Rayleigh fading on the streams | `channel.type=rayleigh --tag rayleigh` for the five key methods | 5 runs |
| Fixed-rate codebooks (is rate nesting free?) | `codec.granularities=[m]`, compare channel-only accuracy | minutes each |
| More seeds for the headline pair | `--methods nested,single_rate --seeds 3,4` | 4 runs |
| Larger adapter / other LoRA targets | `codec.lora` overrides, needs a new `base` | one day |

### C. Known weak points to address or to state

- Shipping pays only at high traffic: the crossover should be discussed as a
  result, with the cheaper-synchronisation variants of the sensitivity table.
- One language model family so far, and a small one.
- Context is the receiver's own decoded text, but refinement and context are
  emulated teacher-forced on the streams; the free-running checks are in the
  probes and in `eval_refine.py`.
- No comparison with a trained end-to-end baseline (DeepSC-style). The
  separation bound and the `none`/`unigram` receivers bracket it, but a
  reviewer may still ask.

## Writing

Suggested structure, mirroring the standards of your earlier papers:

1. Introduction — a semantic link is a shared prior; the prior drifts; the
   receiver cannot adapt alone; adaptation has an air-time price.
2. Related work with a synthesis table of evidential controls.
3. System model: token codebook, Bayesian receiver, net semantic rate.
4. Method: rate nesting; the three memories; slots; selective refinement.
5. Analysis: staleness versus synchronisation cost per level; the crossover.
6. Experimental setup — everything fixed before results.
7. Results in the order of the claims of `docs/METHOD.md`.
8. Discussion: what the evidence supports and what it does not; design rules.
9. Limitations; conclusion.

Topics of the call covered: LLM-enabled semantic communication frameworks;
semantic compression and task-oriented strategies; LLMs at the edge;
security/trust and explainability (calibrated confidence); energy- and
spectrum-efficient solutions; experimental platforms.

Before submission: read the journal's author guidelines for length, template
and the policy on AI-assisted writing, and disclose AI assistance as you did
in the ECG paper.
