# Results with a pretrained LLM (GPT-2, 124M)

Everything below was produced by the code in this repository with
`configs/gpt2.yaml` on two CPU cores. Every number is copied from the
generated tables in [`TABLES.md`](TABLES.md) (which also shows all figures);
the files behind them are in this folder. Nothing was tuned on the test
material, and results that go against the method are listed in Section 9.

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

In words: at equal fidelity the complete system needs about half the channel
symbols of a frozen fixed-rate receiver (4.55 fewer symbols per token). Of
that saving, 71 % comes from selective refinement, 22 % from the context
memory and 7 % from shipped adaptation. The proposed schedule reaches the
per-message upper reference with 2.0 % of its synchronisation bits.

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
(expected calibration error 0.004–0.021 on the base corpus, 0.003–0.006 on the
streams for every method). In the teacher-forced sweep at 10 dB, at the same
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
(`context_only`), 0.609 (`nested`), 0.613 (`full_online`).

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
synchronisation budget, and the interval excludes zero; it is statistically
indistinguishable from shipping after every message. The advantage comes
from the slots (`nested_no_slots` is no better than `single_rate`); giving the
bias its own shorter period contributes little (`nested_two_level`). On the
held-out AG News probe at the end of the stream — after the last TeleQnA
regime — `nested` holds 5.83 bits/token against 6.10–6.14 for the
single-adapter methods: the adapter of the earlier domain was kept.

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
  the stream but treats the seeds as exchangeable.
- **Emulation**: on the streams, refinement and context are scored
  teacher-forced (the receiver's context is its own decoded text, but each
  token is scored given the true tokens before it). Free-running decoding is
  checked in Sections 3–5 (probes) only at fixed prefix lengths and on the
  base corpus for refinement.
- **Synchronisation is charged at channel capacity**, which is optimistic;
  real control signalling would raise every `N*`.
- **One small language model**, and the temporal corpus is a GitHub mirror
  that lacks the first nine months of 2012 (`docs/DATASETS.md`).
- No sentence-embedding similarity (the model could not be downloaded here);
  the decoded texts are stored, so it can be added later.

## 10. Reproduce

```
python -m pip install -e ".[dev]" onnx
python scripts/fetch_gpt2.py --out models/gpt2
python scripts/run_pipeline.py --config configs/gpt2.yaml --stage base
python scripts/run_pipeline.py --config configs/gpt2.yaml --stage streams --seeds 0,1,2
python scripts/run_pipeline.py --config configs/gpt2.yaml --stage report
```

Variant runs: `scripts/run_stream.py --methods nested --seeds 0 --tag <name>
<override>` with `nested.sync.bits=4`; `nested.sync.bits=4 nested.sync.topk=0.1
nested.periods.slow=100`; `nested.lrs.slow=0.0`; `channel.type=rayleigh`.
The trained codebook used for all of the above is `base/codec.pt`; the
per-step logs are in `raw_streams.tar.gz`.
