# Method

Working title: **Doubly Nested Semantic Communication: Rate- and Time-Nested
LLM Receivers for Drift-Resilient 6G Links.**

This document fixes the method and the claims *before* results exist, so that
the results can go against us. Section 8 records what was tried first and
failed.

## 1. Problem

An LLM-based semantic link works because transmitter and receiver share
knowledge. After deployment that shared knowledge drifts, at three speeds:

| Band | What drifts | Timescale |
| --- | --- | --- |
| fast | channel state; the words of the current exchange | per message |
| medium | topic of the conversation | minutes |
| slow | domain vocabulary, world knowledge | days and longer |

A frozen receiver degrades. An adapted receiver has a cost that is almost
always left out: **the receiver cannot adapt on its own, because only the
transmitter knows the source text.** Every receiver-side update is traffic on
the very link the codec is meant to economise.

## 2. System

```
text ─► tokens ─► semantic codebook (M real symbols per token, nested) ─► block power control
                                                   │  send the first m ≤ M symbols of every token
                                             AWGN / Rayleigh
                                                   ▼
text ◄─ MAP decision per token ◄─ posterior ∝ prior(token | decoded so far) × likelihood(received | token)
                                              ▲
                     frozen LLM + three memories (cache, logit bias, LoRA)
```

Both ends hold the same frozen LLM. The transmitter uses only its tokenizer and
embedding table; the receiver uses the LLM as a **language prior**.

### 2.1 Semantic codebook, nested in rate

Every token `v` has a codeword `c_v ∈ R^M`, produced by a small MLP from the
LLM's (frozen) embedding of `v`. The `M` coordinates are cut into shells at the
granularities `m_1 < … < m_K = M`.

- Each shell is centred and scaled to unit average power under the unigram
  distribution of the base corpus, with a per-codeword peak limit.
- Before transmission each shell of each message is rescaled to exactly unit
  power (block power control). The gain is part of the channel estimate, so
  no method can gain power from a domain whose tokens happen to have large
  codewords.
- Consequently every prefix satisfies the power constraint, and sending shell
  `k+1` after shells `1..k` is identical to having sent prefix `m_{k+1}` at
  once: incremental refinement is exact.

The codebook is trained once, without the LLM, by maximising the channel-only
posterior of the transmitted token at randomly drawn prefix lengths and SNRs.
Training takes minutes on a CPU.

### 2.2 Receiver

For token position `t` the receiver computes, over the whole vocabulary,

```
log p(v | y, past) = w_p · log p_LLM(v | decoded tokens before t) + b_v + w_l · log p(y_t | c_v) + const
```

with the exact Gaussian likelihood of the received prefix, and decides the
maximum. Decoding is free-running: the LLM is conditioned on the receiver's
own earlier decisions. Three reference receivers isolate the LLM's
contribution: `llm` (above), `unigram` (token frequencies only) and `none`
(maximum likelihood).

### 2.3 Nested in time: three memories of the prior

| Level | What | Changes | Synchronisation cost |
| --- | --- | --- | --- |
| fast | token cache: decayed counts of the receiver's own recent decisions, mixed into the prior | every message | none |
| medium | `w_p`, `w_l`, cache share, and a logit bias `b = E·u` in embedding space (one vector `u`) | every `C_med` messages | a few kilobits |
| slow | LoRA adapter of the LLM, EMA-anchored | every `C_slow` messages | megabits |

The gradient levels need the true text, so they are learned at the
transmitter, which keeps a *working copy*: it takes one optimiser step per
message. Once per period of a level the transmitter ships the **quantised**
difference between its working copy and the receiver's copy; both ends apply
the quantised difference, so the copies are bit-identical right after a
shipment. Between shipments the receiver decodes with a stale copy. A level's
period therefore trades staleness against air time, and nothing else: all
methods learn in the same way and differ only in *when* each level reaches the
receiver. Every shipped bit is recorded by an exact ledger.

Information pattern, stated once: the transmitter knows the text, the
codebook, the fed-back average SNR and its own working copy. It does not know
the receiver's noise or decisions; it trains on its own simulation of the
channel and mirrors the cache with the true tokens.

(`nested.rule: chunk` switches to the literal chunked rule of the continuum
memory system — gradients accumulated at fixed parameters and applied once
per period — as an ablation. It keeps the two copies equal at all times but
learns `C` times more slowly.)

### 2.4 Net semantic rate

```
R_net = m · T + B_sync / ( C(snr) · N_msg )          real symbols per message
```

`T` is the number of tokens per message, `B_sync` is measured by the ledger,
`C(snr)` is the capacity at which sync bits are assumed to travel (optimistic,
and equally so for every adaptive method), `N_msg` is the traffic carried.
From it,

```
N* = b_sync_per_step / ( C(snr) · Δm · T )
```

is the traffic per adaptation step above which adapting is cheaper than not
adapting and sending `Δm` more symbols per token for the same fidelity. These
are derived quantities and are labelled as such.

### 2.5 Calibrated selective refinement

The posterior gives every token a confidence. The receiver decodes from the
shortest prefix and requests the next shell **only for the tokens whose
calibrated confidence is below a threshold** — a token-level incremental
redundancy scheme driven by the LLM's own uncertainty. Calibration (Platt
scaling on the confidence and the level) is fitted on validation text; its
error is reported before and after. Controls: the fixed-rate curve
(time-sharing between granularities), an oracle that knows which tokens are
wrong, and a random policy with the same request rate.

## 3. Baselines

| Name | What it isolates |
| --- | --- |
| `frozen` | the cost of drift itself |
| `cache_only` | how far free, receiver-only adaptation gets |
| `single_rate` | **the matched ablation**: both gradient levels shipped at one period, chosen so the sync budget equals the nested method's |
| `ewc` | a standard continual-learning regulariser on top of `single_rate` |
| `full_online` | shipped after every message: adaptivity upper reference at prohibitive cost |
| `nested_no_ema`, `nested_no_cache`, `nested_two_level` | each component of the proposal |
| `unigram` / `none` receivers | the value of the LLM prior itself |
| ideal separation: DEFLATE or same-LLM arithmetic coding, with an asymptotic and a finite-block-length (normal approximation) channel code | the strongest classical opponent, as a derived bound |

## 4. Claims, and what would refute each

1. **An LLM prior buys rate.** At equal fidelity the `llm` receiver needs
   fewer symbols per token than `unigram`/`none`. *Refuted if* the curves
   coincide. (Expected to hold; it is the starting point, not the novelty.)
2. **Rate nesting is free**: one nested codebook matches codebooks trained for
   a single prefix length. *Refuted if* single-length training is better
   beyond a margin fixed in advance (equivalence test).
3. **Time nesting beats single-rate adaptation at equal sync budget** on
   prequential fidelity. *Refuted if* `single_rate` matches or beats `nested`
   — as it might: in your own H-BAYO study pooled training beat meta-learning,
   and in Nested-MARL the EMA anchor was dormant.
4. **The gain needs hierarchical drift**: it should shrink on streams without
   timescale separation. *Tested by* varying `tau_med`, `tau_slow`, `kappa`,
   and by the `temporal` stream.
5. **Adaptation has a crossover**: below a traffic volume `N*` a frozen
   receiver with a longer prefix is the better engineering choice. This is a
   result whichever side a deployment falls on.
6. **Uncertainty-triggered refinement saves symbols at equal fidelity**
   against fixed rates and beats the random control. *Refuted if* it does not
   beat the random control.

Reporting rule: whichever way each falls, it is reported.

## 5. Relation to prior work (verify each before citing)

- **Nested Learning** (Behrouz et al., NeurIPS 2025) — the multi-frequency
  update principle. Not previously applied to a transceiver, where update
  frequency acquires a cost in channel uses.
- **LLM-SC** (Wang et al., arXiv 2407.14112) — tokens as semantic units and
  the pretrained LLM as the prior of an optimal decoder, with beam search.
  This is the closest base system and must be cited as such. It is
  fixed-rate, frozen, and does not consider drift or update cost.
- **Token Communications** (Qiao et al., arXiv 2502.12096) — large-model
  context among tokens for bandwidth efficiency; no nesting, no adaptation
  cost.
- **Matryoshka Representation Learning** (Kusupati et al., NeurIPS 2022) —
  prefix nesting, not as a channel-rate mechanism with exact power control.
- **DeepSC** (Xie et al., 2021) and successors — fixed-rate, stationary.
- **Generative Semantic HARQ** (arXiv 2603.15068) — retransmission by
  stochastic re-encoding; ours sends nested refinements for selected tokens
  under a calibrated trigger.
- **Knowledge-base mismatch / update** (e.g. Li et al., arXiv 2410.02405;
  transceiver cooperative learning against mismatched knowledge bases) —
  update overhead is treated qualitatively, at one timescale, without an LLM.

The novelty search behind this list was a handful of web queries, not a
systematic review. A proper sweep (IEEE Xplore, arXiv, Scopus) is a
prerequisite for the claim "first", and is scheduled in the roadmap.

## 6. Theory to write (after results)

Because every method learns identically and differs only in the shipping
periods, the receiver's excess loss at a level is a staleness term that grows
with `C_l` times the drift rate of what the level tracks, while its cost is
`b_l / C_l`. Minimising their sum per level suggests a square-root law,
`C_l* ∝ sqrt(b_l / (drift rate_l · traffic))`: cheap, fast-drifting levels
should be shipped often and expensive, slow-drifting ones rarely — the nested
schedule is then a consequence, not an assumption. Your two-timescale
convergence argument from Nested-MARL supplies the tracking part, with
quantisation as bounded noise. This is a plan, not a result; the period sweep
in the roadmap is its empirical check.

## 7. Known limitations of the design

- Simulated channel; no over-the-air measurement; perfect channel state.
- Uncoded symbol-level transmission of tokens; no outer channel code.
- The sentence length is assumed known at the receiver.
- Greedy free-running decoding; beam search would help at the lowest rates.
- Text only; the call also invites multimodal work.
- Sync bits are charged at capacity, which understates real overhead.
- The transmitter must run the LLM to learn the receiver's prior, so
  adaptation costs transmitter compute; this is measured.
- Small LLMs (≤ 0.5 B parameters) on a CPU, by hardware necessity.

## 8. What was tried first (negative result, kept on purpose)

Version 0.1 (git tag `v0.1-sentence-codec`) compressed a whole sentence into
8 memory tokens and regenerated it from soft prompts (an ICAE-style
autoencoder over the channel). After 13.6 h of CPU pretraining with
SmolLM2-135M it reached a teacher-forced token accuracy of 0.41–0.44 and
BLEU-4 ≈ 0.01, while ideal separation at the same budget delivers almost
every sentence. Sentence-level soft-prompt autoencoding needs far more
training than a CPU allows, and at that scale it loses to classical coding.
The token-level design above replaces it: the LLM is used for what a small
model already does well without training — predicting the next token.
