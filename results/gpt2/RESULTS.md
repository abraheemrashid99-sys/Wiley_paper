# Results with a pretrained LLM (GPT-2, 124M) — version 0.2 result set (pilot)

Everything below was produced by the code in this repository with
`configs/gpt2.yaml` on two CPU cores. Every number is copied from the
generated tables in [`TABLES.md`](TABLES.md) (which also shows all figures);
the files behind them are in this folder. Nothing was tuned on the test
material, and results that go against the method are listed in Section 9.

> **How to read this document after the review** (`docs/REVIEW.md`). These
> numbers are a pilot, not the paper's results. Four limits apply to all of
> them and are detailed in Section 9:
>
> 1. **Three seeds cannot show significance at the seed level.** The exact
>    seed-level test in `TABLES.md` gives p = 0.25 or more before and 1.000
>    after Holm correction for every comparison; 0.25 is the floor for three
>    seeds. The
>    intervals quoted below are block-bootstrap intervals over steps.
> 2. **Symbols per token are forward symbols only.** The return-link
>    signalling of selective refinement — which supplies 71 % of the headline
>    saving — is not charged in this result set.
> 3. **Refinement on the streams is emulated teacher-forced**, not decoded for
>    real.
> 4. **The 37 stream runs were produced by five code revisions** and were
>    never re-run from scratch as a whole here (they were on a GPU, from one
>    revision: Section 12).
>
> Version 0.3 of the code measures items 2 and 3 and fixes the protocol for
> items 1 and 4 (`docs/METHOD.md`, section 9). Section 11 reports what was
> re-run to check this folder, and Section 12 an independent reproduction of
> all of it on a GPU with the new measurements on every seed: **decoded for
> real and with the signalling charged, the full system saves about 26 %
> (hierarchical stream) and 21 % (temporal stream) of the symbols of a frozen
> fixed-rate receiver, not 47.7 % and 38 %.** Section 13 reports the decoder
> chosen for version 0.4 on validation text and a first look at it on one
> test seed (27.9 % decoded for real, and an emulation that now holds within
> 5 %).

## 1. Setup in brief

| Item | Value |
| --- | --- |
| Language prior at the receiver | GPT-2 small (124M parameters), frozen |
| Codebook | 16 real symbols per token, nested at 2/4/6/8/12/16; trained once (37 min), 213 k parameters |
| Receiver memories | context ≤ 48 tokens of the same document; logit bias (771 parameters); 2 LoRA slots (rank 8, last 6 layers, 147 k parameters each) |
| Synchronisation | bias every 10 steps, active LoRA slot every 50 steps, 8-bit quantised differences, exact bit ledger |
| Channel | AWGN (Rayleigh block fading where stated), SNR per step from a clipped AR(1) process, mean 10 dB, range 0–20 dB |
| Base corpus | Europarl sample (NLTK), 14 006 sentences of 4–30 words |
| Hierarchical stream | AG News → TeleQnA → AG News → TeleQnA, 150 steps each, 6 parallel conversations, topic switch every 30 steps; 600 steps, 3 seeds |
| Temporal stream | HuffPost news in chronological order (Oct 2012 – Sep 2022, GitHub mirror), 600 steps, 2 seeds |
| Protocol | prequential: every message is scored with the copy of the parameters the receiver holds, before any update |
| Statistics | mean ± SD over seeds; per-step paired differences with a 95 % block-bootstrap interval (25-step blocks pooled over seeds) |

Methods are defined in [`docs/METHOD.md`](../../docs/METHOD.md), Section 3.

## 2. Headline: what 90 % token accuracy costs in channel symbols

Hierarchical stream, three seeds. "Symbols/token" is read off each method's
measured accuracy-versus-symbols curve.

| Receiver | Prior bits/token | Symbols/token | vs. fixed rate | vs. frozen + refinement | Sync kbit/step |
| --- | --- | --- | --- | --- | --- |
| Frozen, fixed prefix length | 6.793 | 9.550 | — | — | 0 |
| Frozen, selective refinement | 6.793 | 6.315 | −33.9 % | — | 0 |
| + context memory (`context_only`) | 5.627 | 5.323 | −44.3 % | −15.7 % | 0 |
| + single-rate adaptation (`single_rate`) | 5.362 | 5.077 | −46.8 % | −19.6 % | 23.7 |
| + nested adaptation, one adapter (`nested_no_slots`) | 5.349 | 5.063 | −47.0 % | −19.8 % | 24.2 |
| **+ nested adaptation with slots (`nested`, proposed)** | **5.256** | **4.997** | **−47.7 %** | **−20.9 %** | **24.2** |
| Shipped after every message (`full_online`) | 5.251 | 4.977 | −47.9 % | −21.2 % | 1185.8 |

In words: at equal fidelity the complete system needs about half the
*forward* channel symbols of a frozen fixed-rate receiver (4.55 fewer symbols
per token). Of that saving, 71 % comes from selective refinement, 22 % from
the context memory and 7 % from shipped adaptation. The proposed schedule
reaches the per-message upper reference with 2.0 % of its synchronisation
bits. The refinement rows do not include the return-link signalling of the
requests; the fixed-prefix row needs none, so the first comparison is
optimistic by that amount (see Section 11 for a first measurement).

## 3. The LLM prior buys rate (base corpus, free-running decoding)

BLEU-4 on 200 held-out Europarl sentences (195 with a document history),
AWGN, 10 dB:

| Symbols/token | LLM + true context | LLM | Unigram | No prior |
| --- | --- | --- | --- | --- |
| 4 | 0.310 | 0.184 | 0.051 | 0.035 |
| 6 | 0.782 | 0.706 | 0.426 | 0.381 |
| 8 | 0.961 | 0.927 | 0.815 | 0.767 |
| 12 | 0.999 | 0.999 | 0.997 | 0.995 |

Under Rayleigh block fading at 10 dB and 8 symbols/token the same four
receivers reach 0.692 / 0.653 / 0.543 / 0.488. The ideal separation reference
(arithmetic coding with the same LLM plus a finite-block-length code; a
bound, not a system) delivers 42 % of the messages intact at 4 symbols/token
and 96.5 % at 6 (AWGN, 10 dB). The uncoded token scheme does not beat that
bound at a fixed rate (85.7 % token accuracy at 6 symbols); what it offers
instead is graceful degradation and, with selective refinement, 99.1 % token
accuracy at 6.57 symbols (Section 4).

**Is rate nesting free?** A codebook trained for 8 symbols per token only
(same training budget) reaches a channel-only token accuracy of 0.8890 at
10 dB; the nested codebook, read at its 8-symbol prefix, reaches 0.8885
(2 000 held-out tokens; `fixed8/pretrain_summary.json` and
`base/pretrain_summary.json`). At this prefix length nesting costs nothing
measurable; other lengths were not tested.

## 4. Selective refinement and calibration (base corpus)

Free-running, multi-round decoding at 10 dB:

| Channel | Policy | Symbols/token | Token accuracy | BLEU-4 |
| --- | --- | --- | --- | --- |
| AWGN | fixed, 6 symbols | 6.00 | 0.857 | 0.706 |
| AWGN | fixed, 8 symbols | 8.00 | 0.971 | 0.927 |
| AWGN | selective, confidence 0.7 | 5.94 | 0.962 | 0.893 |
| AWGN | selective, confidence 0.9 | 6.57 | 0.991 | 0.978 |
| Rayleigh | fixed, 8 symbols | 8.00 | 0.758 | 0.653 |
| Rayleigh | fixed, 16 symbols | 16.00 | 0.935 | 0.906 |
| Rayleigh | selective, confidence 0.9 | 7.68 | 0.982 | 0.955 |

The fixed-rate rows are from the rate evaluation (200 sentences), the
selective rows from the refinement evaluation (64 sentences of the same
held-out sample).

The receiver's token confidence is calibrated without any correction
(expected calibration error 0.004–0.021 on the base corpus; on the streams
0.003–0.006 as per-method means and 0.002–0.009 over the single runs). In the teacher-forced sweep at 10 dB, at the same
mean number of symbols, confidence-triggered requests beat both controls:
AWGN, 7.13 symbols/token — 1.000 token accuracy against 0.950 for time-sharing
between fixed rates and 0.907 for random requests; Rayleigh, 8.14
symbols/token — 0.931 against 0.848 and 0.762. Under fading, every refinement
round sees a new channel realisation, so part of the gain there is time
diversity, as in HARQ.

## 5. Drift, and what each memory recovers

Prior cross-entropy (bits/token) while a domain is active, three seeds:

| Receiver | AG News | TeleQnA |
| --- | --- | --- |
| `frozen` | 6.597 | 6.990 |
| `context_only` | 6.209 | 5.044 |
| `single_rate` | 6.060 | 4.664 |
| `nested_no_slots` | 6.053 | 4.646 |
| `nested` | 5.968 | 4.544 |
| `full_online` | 5.962 | 4.540 |

Free-running BLEU-4 on held-out probes at 6 symbols/token and 10 dB, averaged
over both domains at the end of the stream: 0.351 (`frozen`), 0.500
(`context_only`), 0.623 ± 0.031 (`single_rate`), 0.613 ± 0.034
(`nested_no_slots`), 0.609 ± 0.011 (`nested`), 0.613 ± 0.034 (`full_online`).
The probes decode 16 sentences per domain, so this metric separates the
receivers with memory from those without and nothing more: among the adapted
receivers the proposal is not the highest, and the differences are within one
standard deviation over seeds.

Per-step paired differences to `nested` (positive = worse than `nested`):

| Method | Seeds | Prior bits/token | Symbols/token at confidence 0.9 |
| --- | --- | --- | --- |
| `frozen` | 3 | +1.538 [+1.315, +1.757] | +1.161 [+0.961, +1.362] |
| `context_only` | 3 | +0.371 [+0.319, +0.421] | +0.307 [+0.239, +0.375] |
| `single_rate` | 3 | +0.106 [+0.078, +0.139] | +0.074 [+0.054, +0.098] |
| `nested_no_slots` | 3 | +0.093 [+0.062, +0.129] | +0.071 [+0.049, +0.096] |
| `ewc` | 1 | +0.129 [+0.078, +0.193] | +0.121 [+0.076, +0.169] |
| `nested_two_level` | 1 | +0.018 [+0.011, +0.026] | +0.012 [+0.004, +0.020] |
| `nested_no_context` | 1 | +1.147 [+0.823, +1.451] | +0.928 [+0.652, +1.210] |
| `full_online` | 3 | −0.005 [−0.031, +0.019] | −0.004 [−0.026, +0.017] |

Reading: `nested` is better than single-rate adaptation at the same
synchronisation budget, and the block-bootstrap interval excludes zero; at
the seed level the difference has the same sign in all three seeds, which
three seeds cannot make significant (p = 0.25). The interval of the
difference to shipping after every message contains zero; that is the
absence of a detected difference, not a demonstrated equivalence (no margin
had been fixed). The advantage comes from the slots (`nested_no_slots` is no
better than `single_rate`); giving the bias its own shorter period
contributes little (`nested_two_level`, one seed). On the held-out AG News
probe at the end of the stream — after the last TeleQnA regime — `nested`
holds 5.83 bits/token against 6.14 for `nested_no_slots`: the adapter of the
earlier domain was kept. These two are compared at equal freshness (both
had just been synchronised at the probe). `single_rate` scores 6.10 on the
same probe, but its copy was 12 steps old at that moment (period 49), so
that figure is not comparable with the other two.

Rows with one seed are comparable only through the paired column, not through
their means in `TABLES.md`.

## 6. Natural drift (temporal stream, two seeds)

| Receiver | Prior bits/token | Symbols/token for 90 % | vs. frozen + refinement |
| --- | --- | --- | --- |
| Frozen, fixed prefix length | 6.331 | 9.130 | — |
| Frozen, selective refinement | 6.331 | 6.007 | — |
| `context_only` | 6.119 | 5.827 | −3.0 % |
| `single_rate` | 5.857 | 5.651 | −5.9 % |
| `nested` | 5.852 | 5.642 | −6.1 % |

The frozen prior degrades with calendar time (6.21 bits/token in the first
era, 6.47 in the last) while the adapted receivers stay near 5.8–5.9. But
`nested` and `single_rate` are practically equal here (+0.005 bits
[+0.002, +0.008]; symbols +0.003 [−0.003, +0.008]): without domains that
return, slots have nothing to recall. The context memory also helps less
(a headline and its summary). `full_online` was run for one seed only (5.845
bits/token on seed 0, against 5.859 for `nested` on the same seed).

### Rayleigh block fading on the stream (one seed)

Same stream, messages and seed, with every message seeing one fading
realisation for all of its shells (no diversity between refinement rounds
here, unlike Section 4):

| Receiver | Prior bits/token | Token accuracy at 6 symbols | Symbols/token for 90 % (selective) | (fixed prefix) |
| --- | --- | --- | --- | --- |
| `frozen` | 6.843 | 0.671 | not reached (0.894 at 10.4 symbols) | not reached (0.896 at 16) |
| `context_only` | 5.832 | 0.723 | 8.62 | 14.71 |
| `nested` | 5.427 | 0.744 | 7.87 | 13.71 |

The ordering of the receivers is unchanged, and the memories matter more
than on the AWGN stream: without them the 90 % target is out of reach within
16 symbols per token.

## 7. When does shipping pay? (derived)

Net semantic rate = payload at equal fidelity + synchronisation amortised over
the messages `N` carried per adaptation step, at 10 dB. `N*` is the traffic
above which a method is cheaper than the context-only receiver, which ships
nothing.

| Configuration (hierarchical stream) | Prior bits/token | Symbols/token for 90 % | Sync kbit/step | N* vs. `context_only` |
| --- | --- | --- | --- | --- |
| `full_online` (3 seeds) | 5.251 | 4.977 | 1185.8 | 110 411 |
| `single_rate` (3 seeds) | 5.362 | 5.077 | 23.7 | 3 101 |
| `nested` (3 seeds) | 5.256 | 4.997 | 24.2 | 2 394 |
| `nested`, seed 0 | 5.284 | 4.829 | 24.2 | 2 735 |
| — 4-bit differences, seed 0 | 5.284 | 4.830 | 12.1 | 1 374 |
| — 4 bit, top 10 %, slow period 100, seed 0 | 5.418 | 4.941 | 3.3 | 614 |
| — bias only (no LoRA), seed 0 | 5.546 | 5.031 | 0.63 | 244 |

Shipped adaptation pays only for links that carry thousands of messages per
adaptation step in the same domain; below that, context and refinement alone
are the better design. Halving the quantiser resolution costs nothing; the
cheap level alone (the logit bias) pays at a tenth of the traffic. Across
methods one bit of prior cross-entropy is worth 0.85–0.88 symbols per token
(`docs/ANALYSIS.md`, Section 2).

## 8. What each memory could buy at best (diagnostic, oracle inputs)

Prior bits/token on 64 held-out sentences per corpus, after 150 adaptation
steps where applicable:

| Corpus | Frozen | + context | + adapted bias | + adapted bias and LoRA |
| --- | --- | --- | --- | --- |
| TeleQnA | 7.269 | 4.843 | 4.581 | 4.024 |
| AG News | 6.923 | 6.600 | 6.410 | 6.081 |
| HuffPost | 6.515 | 6.334 | 6.214 | 5.940 |
| Europarl (base) | 5.791 | 5.137 | — | — |

An adapter fitted to TeleQnA makes the prior for AG News *worse* than no
adapter (6.979 against 6.600) and likewise for HuffPost (6.726 against 6.334):
this negative transfer is what the slots are for.

## 9. What the evidence does not support, and caveats

- **The gain of shipped adaptation is small** next to refinement and context:
  about 6 % fewer symbols on the hierarchical stream and 3 % on the temporal
  one, beyond what the free memories give. The paper should be framed around
  the whole economy (free memories first, shipping above a traffic threshold),
  not around adaptation as the main result.
- **Time nesting as such is not what wins.** At equal budget the benefit over
  single-rate adaptation comes from the slots, not from separate periods, and
  it disappears when domains do not return (temporal stream).
- **EWC did not help** (one seed): it was worse than `nested` and no better
  than `single_rate`.
- **The LLM prior hurts at very low SNR with greedy decoding**: at 0 dB and up
  to 8 symbols/token, free-running decoding with the LLM is below the unigram
  receiver (token accuracy 0.119 against 0.196 at 8 symbols), because early
  errors mislead the prior. Everything fails there (accuracy under 0.2), but
  beam search or a prior weight tied to the SNR is needed before claiming
  robustness at low SNR.
- **Few seeds**: three (hierarchical) and two (temporal). The intervals are
  bootstrap intervals over blocks of steps, which captures variability along
  the stream but treats the seeds as exchangeable. The exact seed-level
  sign-flip test cannot go below p = 0.25 with three seeds; after Holm
  correction every comparison in `TABLES.md` reads 1.000. No claim of this
  document is significant at the seed level.
- **The signalling of selective refinement is not charged.** After each round
  the receiver must tell the transmitter which tokens to continue; with six
  shells that is up to five decisions per token. The symbols-per-token
  figures of the refinement rows are forward symbols only, while the
  fixed-prefix reference needs no signalling.
- **Mixed code revisions.** The stream runs carry five different code
  revisions (11, 9, 8, 6 and 3 runs), because the code was still being
  optimised while they were launched; only `context_only` and `nested`
  (seed 0) were re-run afterwards. Section 11 reports 6 runs repeated with version 0.3.
- **Probe comparisons depend on the shipping phase.** The end-of-stream
  probes fall on steps at which the nested methods have just synchronised,
  while `single_rate` (period 49) is 12 steps stale.
- **Emulation**: on the streams, refinement and context are scored
  teacher-forced (the receiver's context is its own decoded text, but each
  token is scored given the true tokens before it). Free-running decoding is
  checked in Sections 3–5 (probes) only at fixed prefix lengths and on the
  base corpus for refinement.
- **Synchronisation is charged at channel capacity**, which is optimistic;
  real control signalling would raise every `N*`.
- **One small language model**, and the temporal corpus is a GitHub mirror
  that lacks the first nine months of 2012 (`docs/DATASETS.md`).
- No sentence-embedding similarity (the model could not be downloaded here).
  The decoded texts of this result set were not exported, so the rate
  evaluation has to be run again to add it; from version 0.3 on they are
  exported and `scripts/add_similarity.py` adds the metric without a re-run.

## 10. Reproduce

```
python -m pip install -e ".[dev,gpt2]"
python scripts/fetch_gpt2.py --out models/gpt2
python scripts/reproduce.py --config configs/gpt2.yaml --out runs/gpt2_published \
    --export results_repro/gpt2_published --reference results/gpt2 --profile published
```

This runs every experiment behind this folder — the 37 stream runs
(including the variants `bits4`, `lean`, `biasonly`, `lr3x`, `rayleigh`),
the evaluations on the base corpus and the fixed-rate codebook — on one code
revision, and writes `results_repro/gpt2_published/COMPARISON.md`: the
published and the reproduced value of every number, a tolerance check of the
headline numbers and a check of each conclusion. `--profile core` and
`--profile paper` run the confirmatory protocol and the sensitivity studies
(`docs/METHOD.md`, section 9). `notebooks/reproduce_gpt2_colab.ipynb` runs
the same command on a Colab GPU.

The trained codebook used for all of the above is `base/codec.pt`. It was
not part of the version 0.2 commit; it was regenerated for version 0.3 by
running the published training again, which reproduced the published
training log line by line (all 45 logged losses and accuracies identical).
The per-step logs of the stream runs are in `raw_streams.tar.gz`.

Repeatability on the same kind of machine with the same library versions
(`requirements-lock.txt`) is bit for bit: see Section 11. Across machines,
and between a CPU and a GPU, floating-point results differ in the last
digits and the online adaptation carries such differences along, so
agreement there is a matter of tolerance, which is what `COMPARISON.md`
reports.

## 11. Check of this folder with the corrected code (version 0.3)

Made on the same kind of machine and with the same library versions as the
published runs, from one clean code revision (`a25f99d`), with the published
codebook (regenerated: Section 10). The files are in
[`../gpt2_v0.3_check`](../gpt2_v0.3_check).

**Reproduction.** 6 of the 37 stream runs were repeated: `hier_context_only_s0`, `hier_frozen_s0`, `hier_nested_s0`, `hier_single_rate_s0`, `temporal_frozen_s0`, `temporal_nested_s0`.
All 6 reproduce the published files **bit for bit**: compared value by
value with the published per-step logs (`raw_streams.tar.gz`) — every step's
accuracies and cross-entropies at every prefix, the selective curves, the
probes, the slot choices, the synchronisation ledger and the summaries —
116,684 stored values were compared and
116,430 are identical. The 254 that differ are all
topic labels of the temporal stream: a display label whose tie-break
depended on Python's string hashing (made deterministic in version 0.3; no
number depends on it). The corrected code therefore changes nothing that
version 0.2 measured; what follows are additional measurements on the same
runs. The other 31 runs were not repeated here; `--profile published`
repeats all of them.

**New measurements.** They are from seed 0 only and are a first look, not a
result of the paper.

**Hierarchical stream, seed 0 — teacher-forced on all 600 steps (the measurement of version 0.2, now with the signalling of the requests).** Symbols per token for 90 % token accuracy.

| Receiver | Fixed prefix | Selective, forward only | + signalling (entropy) | + signalling (bitmap) | Request bits/token at confidence 0.9 (bitmap / entropy) | Rounds per message |
| --- | --- | --- | --- | --- | --- | --- |
| `frozen` | 9.089 | 6.038 | 7.435 | 7.639 | 3.18 / 2.52 | 5.11 |
| `context_only` | 7.872 | 5.114 | 6.417 | 6.498 | 2.85 / 2.47 | 4.93 |
| `single_rate` | 7.754 | 4.911 | 6.183 | 6.245 | 2.77 / 2.44 | 4.87 |
| `nested` | 7.664 | 4.829 | 6.087 | 6.144 | 2.74 / 2.43 | 4.85 |

**Hierarchical stream, seed 0 — decoded for real (free-running) on the 12 audited steps (72 messages per run).** Same symbols, same target. "Emulated" is the teacher-forced figure on exactly those steps.

| Receiver | Fixed prefix, real | (emulated) | Selective forward, real | (emulated) | Selective + signalling (entropy), real | Selective + signalling (bitmap), real |
| --- | --- | --- | --- | --- | --- | --- |
| `frozen` | 8.531 | 7.579 | 7.111 | 5.337 | 8.427 | 9.036 |
| `context_only` | 7.832 | 7.039 | 6.201 | 4.617 | 7.500 | 7.909 |
| `single_rate` | 7.862 | 7.008 | 5.959 | 4.537 | 7.281 | 7.607 |
| `nested` | 7.634 | 6.831 | 5.709 | 4.407 | 7.004 | 7.300 |

**Temporal stream, seed 0 — teacher-forced on all 600 steps (the measurement of version 0.2, now with the signalling of the requests).** Symbols per token for 90 % token accuracy.

| Receiver | Fixed prefix | Selective, forward only | + signalling (entropy) | + signalling (bitmap) | Request bits/token at confidence 0.9 (bitmap / entropy) | Rounds per message |
| --- | --- | --- | --- | --- | --- | --- |
| `frozen` | 8.813 | 5.835 | 7.224 | 7.390 | 3.15 / 2.52 | 5.04 |
| `nested` | 8.044 | 5.514 | 6.869 | 6.993 | 3.00 / 2.50 | 4.91 |

**Temporal stream, seed 0 — decoded for real (free-running) on the 12 audited steps (72 messages per run).** Same symbols, same target. "Emulated" is the teacher-forced figure on exactly those steps.

| Receiver | Fixed prefix, real | (emulated) | Selective forward, real | (emulated) | Selective + signalling (entropy), real | Selective + signalling (bitmap), real |
| --- | --- | --- | --- | --- | --- | --- |
| `frozen` | 8.180 | 7.309 | 6.908 | 5.039 | 8.207 | 8.791 |
| `nested` | 7.630 | 6.603 | 6.277 | 4.611 | 7.556 | 8.018 |

What they show, for the hierarchical stream:

- **The signalling is not negligible.** At confidence 0.9 the receiver sends
  about 2.4–2.5 bits per token at the entropy of its requests (2.7–3.2 as an
  explicit bitmap), over about five rounds per message. Charged at capacity,
  the saving of the full system against the frozen fixed-prefix receiver
  shrinks from 46.9 % (as in Section 2, for this seed) to
  33.0 % (entropy) or 32.4 % (bitmap), still
  teacher-forced.
- **The teacher-forced emulation is optimistic.** Decoded for real on the
  audited steps, selective refinement needs 33.2 % (`frozen`) and
  29.6 % (`nested`) more forward symbols than the emulation gives
  on exactly those steps; the fixed prefix needs 12.6 % more
  (`frozen`). The reason is error propagation: at short prefixes the
  receiver's own wrong decisions mislead the language prior (at 2 symbols per
  token the frozen receiver decodes 6 % of the tokens for real against 38 %
  teacher-forced). The gap is far above the 5 % of the decision rule in
  `docs/METHOD.md`, section 9, so **the free-running figures are the ones to
  report**.
- **Like for like, decoded for real and with signalling charged, the full
  system saves 17.9 % (entropy) or 14.4 % (bitmap)**
  against the frozen fixed-prefix receiver decoded for real — not
  46.9 %. Of that, the memories account for 10.5 %
  at a fixed prefix (context alone: 8.2 %), which needs no
  signalling at all. Refinement then changes the cost of the `nested`
  receiver by −8.3 % (entropy) or
  −4.4 % (bitmap), and that of the frozen receiver by
  −1.2 % or +5.9 %: with an
  explicit bitmap it makes the frozen receiver more expensive than a fixed
  prefix.

On the temporal stream the picture is the same: decoded for real with
signalling charged, the full system saves 7.6 % (entropy) or
2.0 % (bitmap) against the frozen fixed-prefix receiver, where
the emulation without signalling gave 37.4 %; of that, the memories
give 6.7 % at a fixed prefix, and refinement changes the
cost of the `nested` receiver by −1.0 % (entropy) or
+5.1 % (bitmap) and that of the frozen one by
+0.3 % or +7.5 %.

These figures rest on one seed and 72 messages (about 1200
tokens) per run, and the refinement policy is the simplest one (first shell
of 2 symbols, greedy decoding, a closed token is never reopened). They say
two things with some confidence: the ordering of the receivers (`frozen`, `context_only`, `single_rate`, `nested`, from most to fewest symbols) is the same under real decoding as in the emulation in every selective column, and for `frozen`, `context_only` and `nested` at the fixed prefix too; there `single_rate` and `context_only` change places, by 0.03 symbols on these 72 messages; and the large refinement
gain of Section 2 was mostly an artefact of emulation and uncharged
signalling. What the paper can claim has to come from the confirmatory runs,
after the decoder has been improved on validation text (`docs/REVIEW.md`,
section F).

## 12. Independent reproduction on a GPU, and the new measurements on every seed

All 37 stream runs, the evaluations on the base corpus and the codebook
(trained again from scratch) were repeated with the code of version 0.3.1
(revision `ed2e023`) on a Google Colab Tesla T4, with
`notebooks/reproduce_gpt2_colab.ipynb` and `--profile published`. The files
are in [`../gpt2_v0.3_colab_t4`](../gpt2_v0.3_colab_t4).

**Reproduction: REPRODUCED.** Every headline number lies within 3 % of the
published value (largest relative difference 1.84 %) and all twelve
conclusions checked by `scripts/compare_results.py` hold. Language-model
quantities agree to three decimals; symbols per token are about 1 % lower
on the GPU, the expected size of floating-point differences between a CPU and
a GPU carried through online adaptation. The version 0.2 numbers of this
folder therefore stand as measurements.

**The new measurements on every seed.** Version 0.3 also measured, in the
same runs, what version 0.2 did not: the return-link signalling of the
refinement requests and the free-running audit (every 47th step decoded for
real, at confidence thresholds 0.5, 0.7 and 0.9). Below, "teacher-forced"
is the measurement of Section 2 on all 600 steps, now with the signalling
charged at its entropy; "real" decodes the same received symbols
free-running on the audited steps, read off the operating curves pooled over
the seeds (36 audited steps or 216 messages per receiver on the hierarchical
stream, 24 or 144 on the temporal one). Every receiver is compared with the
frozen fixed-prefix receiver decoded the same way on the same seeds.
Symbols per token for 90 % token accuracy:

**Hierarchical stream, three seeds.**

| Receiver | Teacher-forced: fixed prefix | Teacher-forced: selective + signalling (entropy) | Real: fixed prefix | vs. frozen | Real: selective + signalling (entropy) | vs. frozen | Real: selective + signalling (bitmap) | vs. frozen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `frozen` | 9.490 | 7.690 | 11.452 | — | 9.734 | −15.0 % | 10.447 | −8.8 % |
| `context_only` | 8.162 | 6.608 | 10.957 | −4.3 % | 9.023 | −21.2 % | 9.544 | −16.7 % |
| `single_rate` | 7.859 | 6.320 | 10.648 | −7.0 % | 8.682 | −24.2 % | 9.126 | −20.3 % |
| `nested_no_slots` | 7.851 | 6.312 | 10.570 | −7.7 % | 8.677 | −24.2 % | 9.111 | −20.4 % |
| **`nested`** | 7.774 | 6.232 | 10.318 | −9.9 % | **8.513** | **−25.7 %** | 8.922 | −22.1 % |
| `full_online` | 7.771 | 6.226 | 10.334 | −9.8 % | 8.471 | −26.0 % | 8.883 | −22.4 % |

**Temporal stream, two seeds.**

| Receiver | Teacher-forced: fixed prefix | Teacher-forced: selective + signalling (entropy) | Real: fixed prefix | vs. frozen | Real: selective + signalling (entropy) | vs. frozen | Real: selective + signalling (bitmap) | vs. frozen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `frozen` | 9.004 | 7.355 | 10.906 | — | 9.275 | −15.0 % | 9.945 | −8.8 % |
| `context_only` | 8.543 | 7.095 | 10.538 | −3.4 % | 8.977 | −17.7 % | 9.568 | −12.3 % |
| `single_rate` | 8.246 | 6.942 | 10.302 | −5.5 % | 8.676 | −20.5 % | 9.216 | −15.5 % |
| **`nested`** | 8.246 | 6.932 | 10.343 | −5.2 % | **8.603** | **−21.1 %** | 9.122 | −16.4 % |

(The two −15.0 % / −8.8 % rows of `frozen` agree by coincidence; both were
recomputed from the per-step logs.)

What they show:

- **The emulation is optimistic on every seed.** On exactly the audited
  steps, decoding for real needs 17 % (`frozen`) to 29 % (`full_online`) more
  symbols than the teacher-forced figure at a fixed prefix, and 31–34 % more
  for selective refinement (hierarchical stream; 28–35 % and 36–40 % on the
  temporal one). The free-running figures are the ones to report (decision rule of
  `docs/METHOD.md`, section 9).
- **Like for like, the full system saves 25.7 % (entropy) or 22.1 %
  (bitmap) on the hierarchical stream and 21.1 % or 16.4 % on the temporal
  one** against the frozen fixed-prefix receiver — about half of the
  teacher-forced, signalling-free 47.7 % and 38 %.
- **Refinement pays when decoded for real, on these seeds.** It lowers the
  cost of each receiver below its own fixed prefix by 15–19 % (entropy) or
  9–14 % (bitmap) on both streams. The first look of Section 11 (seed 0
  only, −1.2 % for `frozen`) was not representative: per seed, refinement
  changes the cost of the frozen receiver by −0.6 % (seed 0) and −19 %
  (seed 1), and on seed 2 it does not reach the target within the audited
  thresholds; `nested` with refinement saves 14 %, 27 % and 30 % against the
  frozen fixed prefix of the same seed.
- **The memories keep their order and their size.** At a fixed prefix,
  decoded for real, they save 4.3 % (context alone) to 9.9 % (`nested`) on
  the hierarchical stream and 3.4–5.5 % on the temporal one, with no
  signalling; the ordering `frozen` > `context_only` > `single_rate` >
  `nested` of Section 2 holds in every real column except `single_rate`
  against `nested` at a fixed prefix on the temporal stream (10.302 against
  10.343). `nested` and `full_online` stay within 0.5 % of each other.

**Uncertainty** (re-analysis of the same runs with the scripts of version
0.4: [`../gpt2_v0.3_colab_t4/reanalysis_v0.4`](../gpt2_v0.3_colab_t4/reanalysis_v0.4),
`tables/realdecode_*.md`; 95 % paired bootstrap intervals over the audited
steps within each seed). Hierarchical stream: `nested` saves 25.7 %
[19.6 %, 32.1 %] against the frozen fixed prefix with the signalling at its
entropy and 22.1 % [15.3 %, 29.1 %] with a bitmap; its memories alone, at a
fixed prefix, 9.9 % [4.5 %, 18.9 %]; refinement adds 17.5 % [7.4 %, 22.0 %]
to `nested`. For the frozen receiver and `context_only` the interval is
unbounded on the costly side: in about 8 % of the draws their selective
curve does not reach 90 % within the audited thresholds. Temporal stream:
`nested` saves 21.1 % [14.8 %, 24.9 %]. These intervals describe the
variability of the audited steps, not of the seeds; with three (two) seeds
the seed-level test cannot go below p = 0.25 (0.5), and every comparison
reads at that floor or above.

Limits of these figures. They rest on 216 (144) audited messages per
receiver; seeds differ a lot (seed 0's audited steps need about 8 symbols
per token at a fixed prefix, seeds 1 and 2 about 12), so per-seed figures
scatter widely and a pooled figure is not the mean of the per-seed ones. With
thresholds stopping at 0.9, 5 of the 26 single-seed curves of selective
refinement in these tables never reach the
target, which is why the curves are pooled before reading (the pooled
curve reaches 90 % for every receiver). Version 0.4 therefore audits every
23rd step at thresholds 0.7, 0.9, 0.95 and 0.99. None of this is
significant at the seed level (three seeds). The decoder is the simplest
one (greedy, first shell of 2 symbols, closed tokens never reopened);
Section 13 reports the decoder chosen on validation text for the
confirmatory runs.

**A correction to the headline of `TABLES.md`** (`docs/REVIEW.md`, C8). The
headline table compared runs with one seed with the mean of the frozen
receiver over all its seeds. Against the frozen receiver of the same seed,
the rows read: `ewc` −45.8 % (printed −48.4 %), `nested_no_context`
−37.1 % (−40.1 %), `nested_two_level` −46.7 % (−49.3 %), and on the temporal
stream `full_online` −37.5 % (−39.6 %). In particular `nested_two_level` does
not beat `nested` (−47.7 %); the paired per-step comparison of Section 5
already showed it slightly worse. The published `TABLES.md` is left as it
was produced; the corrected script prints the matched figures.

## 13. The decoder of version 0.4, and a first look at it on a test stream

**The decoder** was chosen on validation text only, by a rule written down
before it ran (`docs/METHOD.md`, section 2.5; `results/decoder_selection`):
beam search of width 4 (ranking sentences by their joint posterior), a first
round of 6 symbols per token, and no reopening of closed tokens. On 96
validation messages it needs 8.306 symbols per token for 90 % token accuracy
decoded for real with the signalling charged, against 8.971 for the decoder
of version 0.3 (−7.4 %). Beam search also lowers the fixed prefix, from
10.994 to 9.049 symbols per token (−17.7 %): much of what refinement had
recovered was error propagation at short prefixes, which a better decoder
removes for the fixed-rate receiver too. Both sides of every comparison
below use the same decoder.

**First look on the hierarchical stream, seed 0** (`frozen`,
`context_only`, `nested`; files in
[`../gpt2_v0.4_check`](../gpt2_v0.4_check); 26 audited steps, 156 messages
per receiver). Symbols per token for 90 % token accuracy; intervals are 95 %
paired bootstrap intervals over the audited steps.

| Receiver | Teacher-forced (600 steps): fixed prefix | selective + signalling (entropy) | vs. frozen fixed | Real: fixed prefix | Real: selective + signalling (entropy) | vs. frozen fixed (real) | what refinement adds (real) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `frozen` | 9.089 | 7.734 | −14.9 % | 10.253 | 8.430 | −17.8 % [−21.5 %, −5.7 %] | −17.8 % [−21.5 %, −5.7 %] |
| `context_only` | 7.858 | 7.100 | −21.9 % | 8.763 | 7.502 | −26.8 % [−31.8 %, −11.4 %] | −14.4 % [−20.8 %, −2.5 %] |
| **`nested`** | 7.648 | **6.998** | **−23.0 %** | 7.979 | **7.392** | **−27.9 % [−33.7 %, −12.5 %]** | −7.4 % [−19.2 %, +0.3 %] |

What it shows, on this one seed:

- **The emulation now holds.** On the audited steps, selective refinement
  decoded for real needs 3.4–4.3 % more forward symbols than the
  teacher-forced emulation (version 0.3 decoder on the same seed: 29.6–33.2 %,
  Section 11), and the fixed prefix 0.5–7.1 % more (12.6 % for `frozen`
  before). By the decision
  rule of `docs/METHOD.md`, section 9 (at most 5 % for every method), the
  teacher-forced curves would stand as the measurement, with the
  free-running ones next to them (`tables/emulation_check_hier.md`). The
  first round of 6 symbols removes the short prefixes at which the
  receiver's own errors mislead the language model, and beam search
  repairs much of the rest.
- **The signalling is lighter.** At confidence 0.9 the requests cost 1.4–1.6
  bits per token at their entropy (1.5–1.7 as a bitmap), over about three
  rounds per message, against about 2.5 bits and five rounds with the
  version 0.3 decoder.
- **The full system saves 23.0 % teacher-forced and 27.9 % decoded for
  real** against the frozen fixed prefix with the same decoder, signalling
  charged; with the version 0.3 decoder this seed gave 17.9 % decoded for real
  (Section 11, other audited steps). Refinement alone saves the frozen
  receiver 17.8 % (it changed its cost by −1.2 % before), and the memories at
  a fixed prefix 14.5 % (context) and 22.2 % (`nested`).
- **What refinement adds shrinks as the receiver improves:** 17.8 % for the
  frozen receiver, 14.4 % with the context memory, 7.4 % for `nested`, whose
  interval reaches zero. The memories and refinement overlap: both remove
  the uncertainty that refinement would otherwise pay to resolve.

One seed, and one that had been seen before: the confirmatory runs
(`--profile core`, six seeds, both streams) decide what the paper may claim.

## 14. Six seeds with the decoder of version 0.4 (the primary comparison)

Files: [`../gpt2_v0.4_six_seeds`](../gpt2_v0.4_six_seeds) (provenance in its
README; the paper's results section is `paper/results_section.tex`).
Hierarchical stream; `frozen` and `nested` on seeds 0–5, `context_only` on
seeds 0, 3, 4, 5; every receiver decoded with beam 4 from a first round of 6
symbols. Real channel symbols per token for 90 % token accuracy, signalling
charged at its entropy:

| Receiver | Seeds | Teacher-forced (600 steps) | vs. frozen fixed | Decoded for real (156 audited steps per seed) | vs. frozen fixed [95 % CI] |
| --- | --- | --- | --- | --- | --- |
| frozen, fixed rate | 6 | 9.227 | — | 10.366 | — |
| frozen + refinement | 6 | 7.805 | −15.4 % | 8.363 | −19.3 % [−21.4, −15.6] |
| `context_only`, fixed rate | 4 | 8.044 | −10.1 % | 9.052 | −10.6 % [−16.1, −5.3] |
| `context_only` + refinement | 4 | 7.178 | −19.8 % | 7.547 | −25.4 % [−29.0, −15.6] |
| `nested`, fixed rate | 6 | 7.797 | −15.5 % | 8.168 | −21.2 % [−23.9, −14.9] |
| **`nested` + refinement** | 6 | **7.075** | **−23.3 %** | **7.478** | **−27.9 % [−31.7, −21.6]** |

- **The emulation check passes** (real against teacher-forced forward symbols
  on the audited steps: +3.4 %, +3.6 %, +5.0 %), so by the rule of
  `docs/METHOD.md`, section 9, the teacher-forced figures are the
  measurement: **DN-SC saves 23.3 % with signalling**, lower on every seed
  (−10.5 % to −28.5 %; p = 0.031, the floor for six seeds), and 20.6 % on the
  three seeds never run before. Decoded for real it saves 27.9 %.
- The memories alone save 15.5 % at a fixed rate with no signalling (prior
  6.79 → 5.23 bits per token); refinement alone saves the frozen receiver
  15.4 %; on top of the memories it adds 9.3 % (8.4 % real, [3.9, 18.3]).
  Requests cost 1.39 bits per token over 2.84 rounds per message.
- Shipped adaptation over the context memory: −0.37 bits per token of prior
  and −0.12 symbols per token at the operating threshold (per-step intervals
  exclude zero; four seeds, p = 0.125). Its synchronisation (24.2 kbit per
  adaptation step) pays only above about 1,100 messages per step against the
  frozen receiver and 6,100 against the context memory.
- Limits: Holm over the three receivers gives 0.094; `context_only` has four
  seeds; the frozen receiver's gap sits at the 5 % limit; the temporal stream
  was not repeated with this decoder; two machines (same code and codebook).

