# Method

Working title: **Doubly Nested Semantic Communication: Rate- and Time-Nested
LLM Codecs for Drift-Resilient 6G Links.**

This document fixes the method and the claims *before* results exist, so that
the results can go against us.

## 1. Problem

An LLM-based semantic link works because transmitter and receiver share
knowledge. After deployment that shared knowledge drifts, at three speeds:

| Band | What drifts | Timescale |
| --- | --- | --- |
| fast | channel state (SNR, fading) | per message |
| medium | topic of the conversation | minutes |
| slow | domain vocabulary, world knowledge | days and longer |

A frozen codec degrades; a codec fine-tuned at one rate either forgets or has
to ship its receiver update constantly. And shipping is the point almost
always left out: **the receiver cannot adapt on its own, because only the
transmitter knows the source sentence.** Every receiver-side update is traffic
on the very link the codec is meant to economise.

## 2. System

```
sentence ─► LLM (frozen) + tx-LoRA, S memory tokens ─► linear ─► L symbols ─► shell power norm
                                                                     │  send prefix m ≤ L
                                                               AWGN / Rayleigh
                                                                     ▼
sentence ◄─ LLM (frozen) + rx-LoRA ◄─ P soft prompts ◄─ de-projection (+ low-rank adapter)
```

One frozen backbone is held at both ends; all learning is in small adapters.

### 2.1 Nested in rate

The payload is cut into shells at the granularities `m_1 < … < m_K`, and each
shell is normalised to unit average power on its own. Consequences:

- every prefix satisfies the same power constraint exactly;
- sending shell `k+1` after shells `1..k` is identical to having sent prefix
  `m_{k+1}` at once — incremental refinement is exact, not approximate.

Training uses a stochastic nesting objective: each step draws a few
granularities and averages their losses over one encoder pass.

### 2.2 Nested in time

Parameters are grouped into levels with update periods `C_fast ≤ C_medium ≤
C_slow`, following the continuum-memory rule of Nested Learning: gradients
accumulate every step and are applied once per period.

| Level | Where | Period | Sync cost |
| --- | --- | --- | --- |
| `tx_fast` — memory tokens, projection | transmitter | every step | none |
| `tx_lora` — encoder LoRA | transmitter | medium | none |
| `rx_adapter` — low-rank de-projection adapter | receiver | medium | small |
| `rx_lora` — decoder LoRA, EMA-anchored | receiver | slow | large |

Receiver updates are computed at the transmitter (self-supervised: it can
simulate the channel and holds a copy of the receiver adapters), quantised,
and the **quantised** update is applied at both ends. The two copies therefore
stay bit-identical, and between two updates of a level nothing is stale —
staleness is expressed only through the period.

### 2.3 Net semantic rate

```
R_net = m + B_sync / ( C(snr) · N_msg )          real symbols per message
```

`B_sync` is measured exactly by the ledger; `C(snr)` is the capacity at which
sync bits are assumed to travel (optimistic, and equally so for every adaptive
method); `N_msg` is the traffic the link carried. From it:

```
N* = b_sync_per_step / ( C(snr) · Δm )
```

the traffic per adaptation step above which adapting is cheaper than the
alternative of not adapting and sending `Δm` more symbols per message for the
same fidelity. `N*` — equivalently a crossover drift rate — is the number a
system designer should compute first. These are derived quantities and are
labelled as such.

### 2.4 Calibrated incremental refinement

The receiver decodes from the shortest prefix, maps its decode statistics
(mean token log-probability, entropy, minimum token probability) to a
calibrated probability that the reconstruction is acceptable, and requests
the next shell only if that probability is below a threshold. Calibration is
fitted on validation data and its error (ECE) is reported before and after.

## 3. Baselines

| Name | What it isolates |
| --- | --- |
| `frozen` | the cost of drift itself |
| `tx_only` | how far free, transmitter-only adaptation gets |
| `single_rate` | **the matched ablation**: all levels at one period, chosen so the sync budget equals the nested method's |
| `ewc` | a standard continual-learning regulariser on top of `single_rate` |
| `full_online` | adaptivity upper reference at prohibitive sync cost |
| `nested_no_ema`, `nested_two_level` | each component of the proposal |
| fixed-rate codec, same backbone | the cost of rate nesting (train with one granularity) |
| ideal separation: DEFLATE or same-LLM arithmetic coding, with an asymptotic and a finite-block-length (normal approximation) channel code | the strongest classical opponent, as a derived bound; the finite-block-length version is the fair one at 16–256 symbols |

## 4. Claims, and what would refute each

1. **Rate nesting is free**: one nested codec matches per-rate codecs of the
   same backbone. *Refuted if* fixed-rate training is better beyond a margin
   fixed in advance (equivalence test).
2. **Time nesting beats single-rate adaptation at equal sync budget** on
   prequential fidelity and on retention. *Refuted if* `single_rate` matches or
   beats `nested` — as it might: in your own H-BAYO study pooled training beat
   meta-learning, and in Nested-MARL the EMA anchor was dormant.
3. **The gain needs hierarchical drift**: it should shrink on streams without
   timescale separation. *Tested by* varying `tau_med`, `tau_slow` and `kappa`,
   and by the `temporal` stream.
4. **Adaptation has a crossover**: below a traffic volume `N*` a frozen codec
   with a longer prefix is the better engineering choice. This is a result
   whichever side of it a deployment falls on.
5. **Calibrated stopping saves symbols at equal fidelity** against the best
   fixed rate, and beats a random policy with the same stopping distribution.
   *Refuted if* it does not beat the random control.

Reporting rule: whichever way each falls, it is reported.

## 5. Relation to prior work (verify each before citing)

- **Nested Learning** (Behrouz et al., NeurIPS 2025) — the multi-frequency
  update principle. Not previously applied to a transceiver, where update
  frequency acquires a cost in channel uses.
- **Matryoshka Representation Learning** (Kusupati et al., NeurIPS 2022) —
  prefix nesting; not previously evaluated as a channel-rate mechanism for an
  LLM semantic codec, nor with shell-wise power control.
- **DeepSC** (Xie et al., 2021) and successors — fixed-rate, stationary.
- **Generative Semantic HARQ** (Han et al., arXiv 2603.15068, 2026) —
  retransmission by stochastic re-encoding with self-consistency triggering.
  Ours sends nested refinements under a calibrated trigger.
- **SC-LMKB** (Hu et al., arXiv 2604.05504, 2026) — one-time LLM enrichment of
  the knowledge base; no post-deployment drift or sync accounting.
- **Mismatched-knowledge-base semantic communication** (2023) — single-rate
  alignment, no LLM, no sync budget.
- **ICAE / prompt-compression autoencoders** — the soft-prompt
  encode/decode mechanism we build on, without a channel.

The novelty search behind this list was a handful of web queries, not a
systematic review. A proper literature sweep (IEEE Xplore, arXiv, Scopus) is a
prerequisite for the claim "first", and is scheduled in the roadmap.

## 6. Theory to write (after results)

Your two-timescale convergence result from Nested-MARL transfers with one
change of interpretation: the slow iterate is the receiver-side level, whose
step is additionally perturbed by bounded quantisation noise. The target is a
finite-time bound in which the quantiser resolution and the period `C_slow`
appear explicitly, giving a principled trade-off between sync bits and
tracking error. This is a plan, not a result.

## 7. Known limitations of the design

- Simulated channel; no over-the-air measurement.
- Text only; the call also invites multimodal work.
- Sync bits are charged at capacity, which understates real overhead.
- The transmitter must run the decoder to compute receiver updates, so
  adaptation costs transmitter compute; this will be measured.
- Small LLMs (≤ 0.5 B parameters), by hardware necessity.
