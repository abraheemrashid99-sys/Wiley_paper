# Method

Working title: **Doubly Nested Semantic Communication: Rate- and Time-Nested
LLM Receivers for Drift-Resilient 6G Links.**

This document fixes the method and the claims *before* results exist, so that
the results can go against us. Section 8 records what was tried first and
failed. Section 9 is the confirmatory protocol: it was written after the
first result set (version 0.2, `results/gpt2`) had been reviewed
(`docs/REVIEW.md`) and before any run of version 0.3, and it fixes the seeds,
the primary endpoint, the tests and the equivalence margins of the runs the
paper reports.

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
              frozen LLM + three memories (context, logit bias, LoRA slots)
```

Both ends hold the same frozen LLM. The transmitter uses its tokenizer and
embedding table to encode; the receiver uses the LLM as a **language prior**.

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
posterior of the transmitted token at randomly drawn prefix lengths and SNRs;
a quarter of every batch is drawn uniformly from the vocabulary so that tokens
the base corpus never uses are separated too.

### 2.2 Receiver

For token position `t` the receiver computes, over the whole vocabulary,

```
log p(v | y, past) = w_p · log p_LLM(v | context, decoded tokens before t) + b_v + w_l · log p(y_t | c_v) + const
```

with the exact Gaussian likelihood of the received prefix, and decides the
maximum. Decoding is free-running: the LLM is conditioned on the receiver's
own earlier decisions. Three reference receivers isolate the LLM's
contribution: `llm` (above), `unigram` (token frequencies only) and `none`
(maximum likelihood).

### 2.3 Nested in time: three memories of the prior

| Level | What | Changes | Synchronisation cost |
| --- | --- | --- | --- |
| fast | **context**: the messages of the same conversation that the receiver has already decoded, placed in front of the message being decoded | every message | none |
| medium | `w_p`, `w_l` and a logit bias `b = E·u` in embedding space (one vector `u`) | every `C_med` messages | a few kilobits |
| slow | **LoRA slots**: a small number of adapters of the LLM, one active at a time | every `C_slow` messages | megabits |

*Fast.* A stream carries parallel conversations; each sends the sentences of
a document in order. A message that continues a document is tokenised as
running text and decoded with the receiver's own reconstruction of the
earlier messages as context. This is the LLM's native memory and costs
nothing on the air. Its errors propagate: a wrongly decoded message is a
wrong context.

*Medium and slow.* These need the true text, so they are learned at the
transmitter, which keeps a *working copy* and takes one optimiser step per
message. Once per period of a level the transmitter ships the **quantised**
difference between its working copy and the receiver's copy; both ends apply
it, so the copies are bit-identical right after a shipment and the receiver
is stale in between. A level's period therefore trades staleness against air
time, and nothing else: all methods learn in the same way and differ only in
*when* each level reaches the receiver. Every shipped bit is recorded by an
exact ledger.

*Slots.* Adapting to one domain damages the prior for another (measured: see
the headroom table). The slow memory is therefore a set of `K` adapters. Every
`select_every` messages the transmitter evaluates, on the text it is about to
send, the receiver's copy of every adapter and the frozen prior, and signals a
switch when another choice is better by a margin; if the frozen prior is the
best the receiver has and an unused adapter exists, that adapter is started.
Only the active adapter learns. A returning domain finds its adapter intact
— recall costs `⌈log2(K+1)⌉` signalling bits instead of a re-synchronisation —
and the receiver is never left with an adapter that is worse than no adapter.

Information pattern, stated once: the transmitter knows the text, the
codebook, the fed-back average SNR and its own working copies. It does not
know the receiver's noise or decisions; it trains with the *true* earlier
messages as context and on a channel realisation of its own.

(`nested.rule: chunk` switches to the literal chunked rule of the continuum
memory system — gradients accumulated at fixed parameters and applied once
per period — as an ablation. It keeps the two copies equal at all times but
learns `C` times more slowly. An optional token cache and an EMA pull of the
slow level exist as switches; both are off by default.)

### 2.4 Net semantic rate

```
R_net = S · T + B_sync / ( C(snr) · N_msg )          real symbols per message
```

`S` is the number of symbols per token a method needs to reach a target token
accuracy (read off its measured operating curve), `T` the tokens per message,
`B_sync` is measured by the ledger, `C(snr)` is the capacity at which sync
bits are assumed to travel (optimistic, and equally so for every adaptive
method), `N_msg` the traffic carried. From it,

```
N* = b_sync_per_step / ( C(snr) · (S_frozen − S_method) · T )
```

is the traffic per adaptation step above which a method is cheaper than a
receiver without memory. These are derived quantities and are labelled so.

### 2.5 Calibrated selective refinement

The posterior gives every token a confidence. The receiver decodes from the
shortest prefix and requests the next shell **only for the tokens whose
confidence is below a threshold** — a token-level incremental redundancy
scheme driven by the LLM's own uncertainty. On the base corpus the confidence
is calibrated (Platt scaling on the confidence and the level, fitted on
validation text; error reported before and after) and compared with the
fixed-rate curve (time-sharing between granularities), an oracle that knows
which tokens are wrong, and a random policy with the same request rate. On
the streams every method is scored at every prefix length and at every
threshold, so each has a complete symbols-versus-accuracy curve; a better
prior shows up as fewer requested symbols at equal accuracy.

**Refinement is not free on the return link.** After every round the receiver
has to say which tokens it wants more of. The protocol: round `r` delivers
shell `r` of every token that is still open; the receiver answers with one
decision per open token; a token that is not continued is closed for good,
and nothing is said after the last shell. With `K` shells that is at most
`K − 1` decisions per token. Two costs are derived from the decisions
actually taken, in bits per token: an explicit **bitmap** (one bit per open
token and round) and the **entropy** of the decisions of each round (what an
ideal entropy coder of the bitmap would spend; a lower reference, not an
implemented code). Both are converted to real symbols at the capacity of the
mean SNR — the assumption under which synchronisation bits are charged — and
added to the forward symbols. Every refinement result is reported forward
only and with signalling; the fixed-prefix receiver needs none.

*Decoder of the refinement (version 0.4; fixed before any validation result
was seen).* The audits of version 0.3 showed that real decoding needs about a
third more symbols than the emulation, because at short prefixes the
receiver's own wrong decisions mislead the language prior. Three remedies are
compared, on validation data only (`scripts/tune_decoder.py`):

- the shell sent in the first round: 2, 4 or 6 symbols per token;
- reopening: a token closed in an earlier round may be requested again when
  re-decoding with a corrected context lowers its confidence; every token
  with shells left then takes part in, and is charged for, every round's
  decisions;
- beam search instead of greedy decoding in every decoding pass (width 2 or
  4). Beam search changes the receiver, so the fixed-prefix reference is
  decoded with the same beam.

*Validation set.* 32 documents from the validation split of each of AG News,
TeleQnA and the HuffPost mirror (the corpora of the streams; the streams use
the training split and the probes the test split, so none of these sentences
is ever scored elsewhere). From each document one sentence at a random
position is sent, with the true preceding sentences of its document as
context; the frozen GPT-2 prior and the published codebook; one channel
realisation per message, AWGN, SNR drawn from the stream's SNR process
(mean 10 dB). All candidates decode exactly the same received symbols.
Thresholds 0.7, 0.85, 0.95 and 0.99.

*Correction, before any candidate was evaluated.* The first launch drew one
SNR per batch of six messages as 16 consecutive steps of the AR(1) process;
with ρ = 0.95 such a short stretch does not represent the process (it
averaged 5.2 dB). This was noticed from the first output, the fixed-prefix
reference curve, and corrected to one SNR per message drawn independently
from the stationary law of the process (normal, mean 10 dB, SD 4 dB, clipped
to 0–20 dB). Nothing else was changed.

*Selection rule.* The criterion is real symbols per token at 90 % token
accuracy, decoded free-running, with the request signalling charged at its
entropy (Section 2.5). Stage 1 compares the six greedy candidates (first
shell × reopening). Stage 2 adds beam widths 2 and 4 to the winner of stage
1; a beam is adopted only if it lowers the criterion by more than 2 %
(it costs computation). Candidates within 1 % of each other count as tied,
and the tie goes to the one with fewer rounds per message. The selected
decoder is then fixed for every stream run; its result on the validation set
is reported, but the claims of the paper rest on the streams.

*First shell.* Starting at the shortest prefix maximises the number of rounds.
The same policy is therefore also scored with every later shell as the first
one (fewer rounds and less signalling for more forward symbols). The paper's
policy is the shortest first shell; the other rows are exploratory.

*Emulation and audit.* On the streams the policy is emulated teacher-forced
on every message (each token is judged given the true tokens before it; one
LLM pass per step), starting at the decoder's first shell; the tokens it
decides are also what the receiver writes into its memories. Every
`stream.live_every` steps (a prime, so that the audit is not in phase with
any shipping period: 23 from version 0.4, 47 in the runs of versions 0.2 and
0.3) the same received symbols are in addition decoded for real —
free-running, round by round, with the receiver's own decisions as the
language-model input and with the configured decoder — at the thresholds
`stream.live_thresholds` (0.7, 0.9, 0.95 and 0.99 from version 0.4; 0.5, 0.7
and 0.9 before, which too often stopped short of the target accuracy when
decoding for real), and so is every fixed prefix, with the same decoder,
because the reference must be audited like the proposal. The emulated and
the real operating curves on exactly those steps are reported side by side;
the audit draws no random number and changes no other result (tested). Each
audited step keeps its own request decisions, so that the entropy of the
signalling can be recomputed for any subset of the audited steps.

## 3. Baselines

| Name | What it isolates |
| --- | --- |
| `frozen` | no memory at all: the per-message receiver of the literature |
| `context_only` | how far the free memory gets |
| `single_rate` | **the matched ablation**: context, and both gradient levels shipped at one period chosen so the sync budget equals the nested method's; one adapter |
| `ewc` | a standard continual-learning regulariser on top of `single_rate` |
| `full_online` | shipped after every message: adaptivity upper reference at prohibitive cost |
| `nested_no_slots`, `nested_no_context`, `nested_two_level` | each component of the proposal |
| `unigram` / `none` receivers | the value of the LLM prior itself |
| ideal separation: DEFLATE or same-LLM arithmetic coding, with an asymptotic and a finite-block-length (normal approximation) channel code | the strongest classical opponent, as a derived bound |

## 4. Claims, and what would refute each

1. **An LLM prior buys rate.** At equal fidelity the `llm` receiver needs
   fewer symbols per token than `unigram`/`none`. *Refuted if* the curves
   coincide. (Expected to hold; it is the starting point, not the novelty.)
2. **The free memory comes first.** Context alone recovers a large part of
   what drift costs, at zero sync. *Refuted if* `context_only` ≈ `frozen`.
3. **Shipping pays only above a traffic volume.** The gradient levels buy a
   further reduction, but only above a crossover `N*`; below it a receiver
   with context alone is the better engineering choice. This is a result
   whichever side a deployment falls on.
4. **Time nesting beats single-rate adaptation at equal sync budget.**
   *Refuted if* `single_rate` matches or beats `nested` — as it might: in your
   own H-BAYO study pooled training beat meta-learning, and in Nested-MARL the
   EMA anchor was dormant.
5. **Slots prevent the loss on return.** With alternating domains, one
   adapter is dragged back and forth; slots keep both. *Refuted if*
   `nested_no_slots` matches `nested` on the revisits.
6. **The gain needs hierarchical drift**: it should shrink on the `temporal`
   stream, where nothing about the drift is designed.
7. **Uncertainty-triggered refinement saves symbols at equal fidelity**
   against fixed rates and beats the random control. *Refuted if* the saving
   disappears once the return-link signalling is charged (Section 2.5), or if
   free-running decoding does not confirm the emulated curve (Section 9).

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
- On the streams, refinement and context are emulated teacher-forced on every
  step (the receiver's own decisions feed its context, but each token is
  scored given the true tokens before it). Free-running decoding of the same
  symbols is measured on a subset of the steps (Section 2.5); the receiver's
  context is always built from the emulated decisions.
- Return-link signalling is charged at capacity and assumed error-free; the
  delay of several rounds is counted (rounds per message) but not modelled.
- A held-out probe scores the receiver's copy at one instant, so two methods
  are comparable on a probe only at equal staleness. Staleness is recorded
  with every probe; claims about retention rest on methods with the same
  shipping phase (`nested` against `nested_no_slots`).
- Slots are allocated greedily and never recycled.
- Text only; the call also invites multimodal work.
- Sync bits are charged at capacity, which understates real overhead.
- The transmitter must run the LLM to learn the receiver's prior, so
  adaptation costs transmitter compute; this is measured.
- Small LLMs (≤ 0.5 B parameters) on a CPU, by hardware necessity.

## 8. What was tried first (negative result, kept on purpose)

Version 0.1 (commit `cd8932d`) compressed a whole sentence into
8 memory tokens and regenerated it from soft prompts (an ICAE-style
autoencoder over the channel). After 13.6 h of CPU pretraining with
SmolLM2-135M it reached a teacher-forced token accuracy of 0.41–0.44 and
BLEU-4 ≈ 0.01, while ideal separation at the same budget delivers almost
every sentence. Sentence-level soft-prompt autoencoding needs far more
training than a CPU allows, and at that scale it loses to classical coding.
The token-level design above replaces it: the LLM is used for what a small
model already does well without training — predicting the next token.

## 9. Confirmatory protocol (fixed before the runs of version 0.3)

The first result set (`results/gpt2`, version 0.2) is treated as a **pilot**.
Its review (`docs/REVIEW.md`) found that it could not support seed-level
significance (three seeds), that it left the signalling of refinement
uncharged, and that its refinement figures were emulated only. The pilot
informed the choices below; none of them was changed after a version 0.3 run
was inspected. Deviations, if any, are listed in the results document.

**Runs.** `scripts/reproduce.py --profile core`, one code revision, one
machine. Hierarchical and temporal stream: `frozen`, `context_only`,
`single_rate`, `nested` on seeds 0–5; on the hierarchical stream also
`full_online`, `nested_no_slots` and `nested_two_level` on seeds 0–5,
`nested_no_context` and `ewc` on seeds 0–2. Seeds 0–2 repeat the pilot's
streams; seeds 3–5 were never run before. Tables are produced for all six
seeds and, as a check against having tuned on the pilot, for seeds 3–5 alone
(`scripts/aggregate.py --seeds 3,4,5 --to tables_unseen`).

**Primary endpoint.** Real symbols per token at 90 % token accuracy on the
hierarchical stream, read off each method's measured operating curve, **with
the return-link signalling charged at its entropy**; the forward-only and
the bitmap figures are reported next to it. The headline comparison is
against the frozen receiver at a fixed prefix length, which needs no
signalling.

**Emulation check.** The teacher-forced curves are accepted as the
measurement if, on the audited steps, the free-running forward symbols at
the target differ from the emulated ones by at most 5 % for every method of
the headline table. Otherwise the free-running figures replace the emulated
ones in the headline — on both sides of the comparison, selective refinement
and fixed prefix — and the emulated ones are reported as such. The audit
covers 26 of 600 steps per run (156 messages; version 0.3: 13 steps); the
free-running headline is therefore read from the operating curves of the
audited steps pooled over the six seeds, every receiver against the frozen
receiver on the same seeds and steps. Its uncertainty is a 95 % percentile
interval from 2000 paired bootstrap draws of the audited steps within each
seed (a draw in which a curve misses the target counts as an infinite cost),
and the seed-level test (ii) below is applied to the per-seed readings
(`tables/realdecode_*.md`).

*Changes in version 0.4, before any confirmatory run.* The audit was made
denser and its thresholds extended (above), its readings are pooled over the
seeds with paired bootstrap intervals, and the decoder of the refinement is
chosen on validation text (Section 2.5). The three-seed figures that
motivated these changes (`results/gpt2/RESULTS.md`, section 12) come from
seeds 0–2 only; seeds 3–5 remain unseen.

*Disclosure.* While the audit was being implemented, one run (frozen
receiver, seed 0, hierarchical stream) was inspected. It showed a gap far
above 5 % for selective refinement and led to one change of this protocol:
the fixed-prefix reference is audited free-running as well. The decision
rule, the seeds, the tests and the margins were not changed. Further runs of
seed 0 were then made to check the published folder
(`results/gpt2/RESULTS.md`, section 11); they confirm the gap. Seed 0 is
therefore no longer unseen under the new measurements — seeds 3–5 are — and
the decoder of the refinement policy is expected to change before the
confirmatory runs (`docs/REVIEW.md`, section F); any such change must be
chosen on validation text and written into Section 2.5 first.

**Tests.** (i) Per-step paired differences to `nested` with a 95 % block
bootstrap (25-step blocks pooled over seeds), which treats seeds as
exchangeable. (ii) An exact seed-level sign-flip permutation test on the
per-seed means, Holm-corrected over the comparisons with `nested` on a
stream; with six seeds its smallest attainable p-value is 2/64 = 0.031, so a
difference counts as significant at the seed level only if all six seeds
agree in sign. A claim of difference requires both (i) and (ii). (iii)
Claims of "no meaningful difference" (`nested` against `full_online`;
`nested` against `single_rate` on the temporal stream) use two one-sided
tests on the per-step differences: equivalent iff the 90 % block-bootstrap
interval lies inside ±0.05 bits/token (prior cross-entropy) and ±0.05
symbols/token (selective refinement at the operating threshold). The margin
is about 1 % of the prior cross-entropy and half of the smallest effect the
pilot interpreted (0.106 bits between `nested` and `single_rate`). A failed
equivalence test is reported as "not shown", never as a difference.

**What each claim of Section 4 needs.** Claim 2: `context_only` against
`frozen`. Claim 4 (time nesting): `nested` against `nested_two_level` — the
only comparison in which nothing but the shipping periods differs. Claim 5
(slots): `nested` against `nested_no_slots`. The comparison of `nested`
with `single_rate` mixes both and is reported, but it decides neither.
Claim 6: the same comparisons on the temporal stream. Claim 7: selective
refinement with signalling against the fixed prefix, for the frozen
receiver, on both streams.

**Exploratory, and labelled so.** The first-shell table, other target
accuracies, the sensitivity studies of the `paper` profile (one seed each)
and the Rayleigh stream (three seeds, three methods).
