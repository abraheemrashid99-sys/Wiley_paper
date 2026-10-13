# Analysis notes (to be turned into Section 5 of the paper)

These are derivations and back-of-envelope laws that explain the measured
results. They are stated at the level of rigour of a design argument, not of
a theorem; each is paired with the measurement that checks it. Verify every
step before it goes into the manuscript.

## 1. The receiver

For the token at position `t`, with received prefix `y ∈ R^m` and per-symbol
noise precisions `w_j` (after power control and equalisation),

```
p(v | y, past) ∝ p(v | past) · exp( −½ Σ_j w_j (y_j − c_vj)² )
              ∝ p(v | past) · exp( Σ_j w_j y_j c_vj − ½ Σ_j w_j c_vj² ).
```

The decision is the maximiser. Two facts follow without further assumptions.

- *The decoder is calibrated if the prior is.* The posterior is exact given
  the prior and the channel law, so the confidence `max_v p(v | y, past)` is
  the probability of a correct decision whenever `p(v | past)` is the true
  conditional law of the text. Measured (version 0.2 result set): expected
  calibration error of 0.002–0.009 over the single stream runs, 0.003–0.006
  as per-method means (stream table), and 0.004–0.021 on the base corpus
  before any correction (refinement table). Miscalibration is therefore a
  *diagnostic of the prior*, which is why confidence can drive refinement.
- *Nesting is exact.* The exponent is a sum over symbols, so the posterior
  after shells `1..k+1` equals the posterior after shells `1..k` multiplied
  by the likelihood of shell `k+1`: a refinement adds information and never
  requires re-sending.

## 2. Prior bits and channel symbols are exchangeable

The receiver needs roughly `H(v | past)` bits per token from the channel. One
real symbol carries at most `C(snr) = ½ log2(1 + snr)` bits, so a memory that
lowers the prior cross-entropy by `Δh` bits per token should save about

```
ΔS ≈ κ · Δh    symbols per token,     κ ≳ 1 / C(snr).
```

`κ` exceeds `1/C` because tokens are sent uncoded and the SNR varies. It can
be read off the results by dividing the change in "symbols per token for 90 %
accuracy" by the change in "prior bits per token" between two methods on the
same stream. This single number links every memory to the air interface: it
is what lets a gain measured in language-model terms be compared with a cost
measured in synchronisation bits.

## 3. When does shipping an update pay?

A method that ships `b` bits per adaptation step and thereby saves `ΔS`
symbols per token on messages of `T` tokens has net semantic rate

```
R_net(N) = S · T + b / ( C · N )      symbols per message,
```

where `N` is the number of messages the link carries per adaptation step. It
beats a method that ships nothing when

```
N > N* = b / ( C · ΔS · T ).
```

`N*` is the first number a designer should compute. It falls linearly with
the size of the shipped update (`b`: quantiser resolution, sparsity, adapter
rank, period) and with the gain. The free memories (context, selective
refinement) have `b = 0` and pay at any traffic.

## 4. How often should a level be shipped?

All methods learn identically at the transmitter; the period `C_l` of level
`l` only delays what the receiver sees. Suppose the text switches regime
every `τ` steps and that a fresh copy of level `l` is worth `g_l` bits per
token. After a switch the receiver's copy is useless for about `C_l / 2`
steps on average (plus the learning time, which is the same for every
schedule). Per step and per message,

```
cost(C_l) ≈  κ · T · g_l · C_l / (2τ)   +   b_l / ( C · N · C_l ).
             └─ staleness ──────────┘       └─ air time ───────┘
```

Minimising over `C_l` gives a square-root law,

```
C_l* = sqrt( 2 τ b_l / ( κ T g_l C N ) ).
```

- Levels with a small update (`b_l` in kilobits: the logit bias) should be
  shipped often; levels with a large one (megabits: a LoRA adapter) rarely.
  A multi-timescale schedule is therefore a *consequence* of unequal update
  sizes, not an assumption.
- More traffic (`N`) justifies shorter periods; so does faster drift
  (smaller `τ`).
- When `C_l* > τ` the level should not be shipped within a regime at all:
  below the corresponding traffic, slots that are filled once and recalled
  are the only form in which the slow memory is worth having.

Empirical checks. The version 0.2 result set contains none for the period
law: the only run in which nothing but the periods differs,
`nested_two_level`, has one seed (+0.018 bits/token against `nested`). The
checks are part of the reproduction profiles of version 0.3 and have not
been run yet: `nested_two_level` on six seeds (`core`), and the slow and the
medium period halved and doubled (`slow_half`, `slow_double`, `medium_half`,
`medium_double` in `paper`). Until they exist, the square-root law is a
design argument, not a finding.

## 5. Why slots

With a single adapter, returning to an earlier domain costs twice: the
adapter must be re-learned, and until the next shipment the receiver holds an
adapter fitted to the *other* domain, which the headroom table shows to be
worse than no adapter. With `K` slots a return costs `⌈log2(K+1)⌉` signalling
bits and no staleness. The saving per return is about one full shipment plus
`κ T g C_slow / 2` symbols per message of stale decoding; it grows with the
number of returns, which is why the gain of slots appears on the second visit
of every domain and on the retention probes.

## 6. What refinement costs on the return link

Selective refinement with `K` shells asks, after each of the first `K − 1`
rounds, one question per token that is still open: continue or stop. Let
`n_r` be the number of open tokens after round `r` and `q_r` the fraction of
them that continue. Per transmitted token (`N` tokens in total),

```
bitmap bits   = Σ_r n_r / N                       (one bit per open token and round)
entropy bits  = Σ_r n_r · H2(q_r) / N             (ideal coding of each round's bitmap)
```

and, with the return link charged at the capacity `C` of the mean SNR,

```
S_total = S_forward + bits / C        symbols per token.
```

Three consequences guide the design. (i) The cost is paid by every token in
every round it is open, so starting at a very short first shell — which
nearly every token outgrows — spends signalling on decisions whose outcome
is almost certain; their entropy is small, but an explicit bitmap pays a
full bit for each. (ii) Starting at a later shell removes those rounds and
their signalling at the price of forward symbols for the tokens that would
have stopped earlier; which first shell minimises `S_total` is an empirical
question, answered by the first-shell table. (iii) A better prior lowers
both terms: fewer forward symbols and fewer open tokens per round. The
fixed-prefix receiver has `bits = 0`, which is why it is the reference of the
primary endpoint (`docs/METHOD.md`, section 9).

## 7. Limits of these arguments

- `κ` is treated as a constant; it depends on the operating accuracy.
- Learning time is ignored in Section 4; it sets a floor on useful periods.
- Synchronisation and the refinement requests are charged at capacity and
  assumed error-free; a real control channel costs more, which raises `N*`,
  lengthens the optimal periods and favours fewer refinement rounds.
- The regime length `τ` is known here; in practice it has to be estimated,
  e.g. from the slot selector's switches.
