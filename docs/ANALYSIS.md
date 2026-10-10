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
  conditional law of the text. Measured: expected calibration error of
  0.005–0.007 on the streams for every method (stream table), about 0.01 on
  the base corpus (refinement table). Miscalibration is therefore a
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

Checks in the result set: the period variants of the sensitivity table
(`slow25`, `slow100`), and the comparison of `single_rate` (both levels at
one period) with `nested` at the same budget.

## 5. Why slots

With a single adapter, returning to an earlier domain costs twice: the
adapter must be re-learned, and until the next shipment the receiver holds an
adapter fitted to the *other* domain, which the headroom table shows to be
worse than no adapter. With `K` slots a return costs `⌈log2(K+1)⌉` signalling
bits and no staleness. The saving per return is about one full shipment plus
`κ T g C_slow / 2` symbols per message of stale decoding; it grows with the
number of returns, which is why the gain of slots appears on the second visit
of every domain and on the retention probes.

## 6. Limits of these arguments

- `κ` is treated as a constant; it depends on the operating accuracy.
- Learning time is ignored in Section 4; it sets a floor on useful periods.
- Synchronisation is charged at capacity; a real control channel costs more,
  which raises `N*` and lengthens the optimal periods.
- The regime length `τ` is known here; in practice it has to be estimated,
  e.g. from the slot selector's switches.
