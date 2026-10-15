# Roadmap

Target: *International Journal of Communication Systems* special issue on
LLM-enabled semantic communications, deadline **31 March 2027**.

## Where the project stands

| Step | State |
| --- | --- |
| Method fixed and written down (`docs/METHOD.md`) | done |
| Pipeline: codebook, receiver, streams, baselines, statistics, tables, figures | done, tested |
| First complete result set with a pretrained LLM (GPT-2, 124M) | done: `results/gpt2` — now treated as a **pilot** |
| Strict review of that result set (`docs/REVIEW.md`) | done |
| Measurement fixes from the review (signalling of refinement, free-running audit, statistics, artefacts) | done in version 0.3, tested |
| Confirmatory protocol written down (`docs/METHOD.md`, section 9) | done |
| Independent reproduction of the pilot on a GPU (Colab T4, all 37 runs) | done: REPRODUCED, `results/gpt2_v0.3_colab_t4`; new measurements on every seed in `results/gpt2/RESULTS.md`, section 12 |
| Decoder for selective refinement, chosen on validation text (`docs/REVIEW.md`, A7) | done in version 0.4: beam 4, first shell 6 symbols (`results/decoder_selection`) |
| **Confirmatory runs (`--profile core`)** | **next** (Colab notebook, `PROFILE = "core"`) |
| Sensitivity studies (`--profile paper`) | to run |
| Second backbone (SmolLM2-135M, `configs/laptop.yaml`) | to run where Hugging Face is reachable |
| Sentence-embedding similarity | to add where Hugging Face is reachable (`scripts/add_similarity.py`) |
| Trained end-to-end baseline; low-SNR decoding | open (see below) |
| Systematic literature sweep for the "first" claims | to do before writing |
| Manuscript | to write |

## What the pilot already suggests, and what it cannot decide

Read `results/gpt2/RESULTS.md` for the numbers and their limits. In one
paragraph: the three free or cheap mechanisms — selective refinement, the LLM
prior and the context memory — carry most of the gain; shipped adaptation
adds a smaller gain, reaches the per-message upper reference at a few percent
of its synchronisation cost, and pays for itself only above a traffic volume
that the crossover analysis makes explicit. What the pilot could not decide:
whether those differences hold at the seed level (three seeds cannot show
it), how much of the refinement gain survives once its return-link
signalling is charged, and whether free-running decoding confirms the
emulated refinement curve. A first look at the last two, on seed 0, is in
`results/gpt2/RESULTS.md`, section 11: most of the refinement gain does not
survive, the gain of the memories does.

## Remaining experiments, in order

### 0. The decoder comes first — done (version 0.4)

The first runs with the corrected code showed that the simplest refinement
policy loses about half of its emulated gain when it is decoded for real and
its signalling is charged (`results/gpt2/RESULTS.md`, sections 11 and 12).
Version 0.4 chose a better decoder on validation text only, by the rule
written into `docs/METHOD.md`, section 2.5, before it was run: beam search
of width 4 and a first round of 6 symbols per token (7.4 % fewer symbols
with signalling than the version 0.3 decoder on the validation set; the
fixed prefix gains 17.7 % from the beam). It is
in the configurations; `--profile core` uses it, and `--profile published`
keeps the decoder of the published runs.

### A. Confirmatory study with GPT-2 (one command)

```
python scripts/reproduce.py --config configs/gpt2.yaml --out runs/gpt2_core \
    --export results_repro/gpt2_core --reference results/gpt2 --profile core
```

88 stream runs and the decoder selection repeated as a check; on two CPU
cores several days with the version 0.4 audit (beam search on every audited
step), on a Colab T4 roughly 8–10 hours (estimated), resumable from Google
Drive. Its runs carry the names of the published ones, but with the decoder
of version 0.4 they follow another protocol, and its `COMPARISON.md` says so
(the published profile is the check of the pilot, done in
`results/gpt2_v0.3_colab_t4`). Afterwards:

```
python scripts/aggregate.py --config configs/gpt2.yaml out_dir=runs/gpt2_core --seeds 3,4,5 --to tables_unseen
```

gives the same tables on the three seeds the pilot never saw. Decision rules
are in `docs/METHOD.md`, section 9; write the results document only after
reading them again.

### B. Sensitivity studies (`--profile paper`)

The same command with `--profile paper` and its own `--out` adds one-seed
studies of the shipping periods, the quantiser, drift speed, mean SNR,
context length, adapter rank and the three-domain streams (32 more
runs). They are exploratory and reported as such.

### C. Second backbone (needs Hugging Face)

```
python scripts/reproduce.py --config configs/laptop.yaml --out runs/smollm2_core \
    --export results_repro/smollm2_core --profile core
python scripts/add_similarity.py --results runs/smollm2_core
```

SmolLM2-135M with the corpora from their official sources (full Europarl,
the complete HuffPost release). Gates before trusting it: the rate table
must show `llm_ctx` ≥ `llm` > `unigram` ≥ `none` at low rate, and
`context_only` must beat `frozen` on the hierarchical stream. This
configuration has never been run end to end; expect to fix small things.
`configs/main.yaml` (SmolLM2-360M) and `configs/scale.yaml` (Qwen2.5-0.5B)
are the same study on larger models where a GPU is available.

### D. Open items that need new work, not just machine time

| Item | Why it matters | What it takes |
| --- | --- | --- |
| Trained end-to-end baseline (DeepSC-style) | reviewers expect it; the separation bound and the `none`/`unigram` receivers bracket it but do not replace it | a small transformer JSCC trained on the base corpus at matched symbols per token, evaluated with the same metrics |
| Decoding at low SNR | at 0 dB greedy free-running decoding with the LLM is below the unigram receiver | beam search, or a prior weight tied to the SNR and tuned on validation text only |
| Energy | the call asks for energy efficiency; only time, parameters and memory are measured | a power measurement on the target device, or a declared-parameter energy model kept apart from measured tables |
| Slot recycling | slots are allocated greedily and never reused | a replacement rule, tested on the three-domain stream |
| Theory | the square-root law of `docs/ANALYSIS.md` is a design argument | the period studies of B, then the tracking argument |

## Writing

Suggested structure, mirroring the standards of your earlier papers:

1. Introduction — a semantic link is a shared prior; the prior drifts; the
   receiver cannot adapt alone; adaptation and refinement both have an
   air-time price.
2. Related work with a synthesis table of evidential controls.
3. System model: token codebook, Bayesian receiver, net semantic rate.
4. Method: rate nesting; the three memories; slots; selective refinement and
   its signalling.
5. Analysis: staleness versus synchronisation cost per level; the crossover;
   the cost of refinement on the return link.
6. Experimental setup — the confirmatory protocol, fixed before the results.
7. Results in the order of the claims of `docs/METHOD.md`.
8. Discussion: what the evidence supports and what it does not; design rules.
9. Limitations; conclusion.

The title follows the evidence. If the confirmatory runs repeat the pilot —
slots help, separate shipping periods do not — "time-nested" is not what the
paper shows, and the title and Claim 4 are rewritten around memories and
slots rather than around timescales.

Topics of the call covered: LLM-enabled semantic communication frameworks;
semantic compression and task-oriented strategies; LLMs at the edge;
security/trust and explainability (calibrated confidence); energy- and
spectrum-efficient solutions; experimental platforms.

Before submission: read the journal's author guidelines for length, template
and the policy on AI-assisted writing, and disclose AI assistance as you did
in the ECG paper.
