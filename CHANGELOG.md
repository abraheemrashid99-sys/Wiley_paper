# Changelog

## 0.5.0 — the manuscript, and the power of refinement rounds

- **Finding (independent audit of the manuscript).** The stream simulation
  transmits every message once, with block power control over the whole
  message, and reads the refinement rounds off that transmission. A
  refinement round, however, carries shell k only for the requested tokens;
  with the gain of the whole message its average power is not one. Measured
  on held-out text (`scripts/check_refine_power.py`, `results/refine_power`),
  the requested tokens carry on average more energy than the message, so
  the refinement rounds of versions 0.2–0.4 were sent at about 1.15 times
  the allowed power (frozen prior, threshold 0.9). Correcting it raises the
  cost of refinement at 90 % token accuracy by about 1–2 % on that text.
- `decoder.refine_power` (`message` | `subset`): with `subset` every
  refinement round is scaled to unit power over the tokens it carries, in
  the teacher-forced emulation (`dnsc.refine.round_policy`) and in real
  decoding (`progressive_decode(..., power="subset")`); `message` reproduces
  the runs of versions 0.2–0.4 exactly (tested). The default stays
  `message` so that the published runs reproduce; new runs should set
  `subset` (`configs/*.yaml`).
- `paper/`: the manuscript (sections, tables, figures, verified references,
  `numbers.json` with the source of every quoted number). Corrections found
  by the audit are stated in the text: readings above the target at the
  cheapest point of a curve are upper bounds; the emulation check covers the
  refinement policy, and the fixed-prefix receivers show larger gaps; the
  teacher-forced seed-level test is reported with its Holm adjustment.

## 0.4.2 — the six-seed result

- `results/gpt2_v0.4_six_seeds`: the primary comparison on six seeds with the
  decoder of version 0.4 (seeds 3–5 on a laptop with `run_quick.py`); RESULTS
  section 14; the paper's results section and its figure
  (`paper/results_section.tex`, `scripts/paper_figure_endpoint.py`).
- Scripts started by `reproduce.py`, `run_pipeline.py` and `run_quick.py`
  import the package next to them.

## 0.4.1

- `scripts/run_quick.py` (`RUN_ON_LAPTOP.md`, VS Code tasks `DN-SC Q` and
  `Q2`): the runs of the primary endpoint (frozen, `nested`,
  `context_only`, hierarchical stream) on a laptop CPU, in parallel within
  the machine's cores and memory, reference and proposal first, resumable;
  the tables at any time; a package to send; pooling with runs made
  elsewhere. Same method, decoder, measurement and codebook as
  `--profile core`. Tested end to end.
- The environment task on Windows uses the `py` launcher when present.

## 0.4.0 — the refinement decoder, and the real-decoding endpoint

**Finding.** The independent GPU reproduction (Colab T4) of all 37 published
runs is REPRODUCED (largest headline difference 1.84 %). Its new
measurements on every seed: decoded for real and with the signalling
charged, the full system saves 25.7 % (hierarchical stream) and 21.1 %
(temporal) of the symbols of the frozen fixed-prefix receiver, against 47.7 %
and 38 % emulated without signalling (`results/gpt2/RESULTS.md`, section 12). With
the decoder chosen in this release, a first look on one test seed gives
27.9 % decoded for real (17.9 % with the old decoder on that seed), and the
teacher-forced emulation agrees with real decoding within 3–4 % instead of
about 30 % (`results/gpt2_v0.4_check`, RESULTS section 13).

Method
- Decoder of selective refinement chosen on validation text only, by a rule
  written down before it ran (`docs/METHOD.md`, section 2.5;
  `scripts/tune_decoder.py`; `results/decoder_selection`): **beam search of
  width 4, a first round of 6 symbols per token, no reopening** — 8.306
  symbols per token at 90 % token accuracy with signalling, against 8.971
  for the version 0.3 decoder. Beam search (ranking sentences by their joint
  posterior, with the language model's cache reordered per beam) and
  reopening of closed tokens are implemented and tested; the configured
  decoder is used for selective refinement and the fixed-prefix reference
  alike.
- Beam search helps the fixed prefix far more than refinement: with beam 4
  the fixed prefix needs 9.049 symbols per token on the validation set
  instead of 10.994, and refinement with signalling saves 8.2 % against it.
- The first selection ranked beams by summed per-position posteriors; found
  on review before any stream result, corrected, the beam stage repeated
  (decision unchanged; disclosed in `docs/METHOD.md`, section 2.5).
- The emulated policy, and the decisions that feed the receiver's memories,
  start at the decoder's first shell.

Measurement and statistics
- Primary endpoint decoded for real (`tables/realdecode_*.md`): operating
  curves pooled over the seeds, every receiver against the frozen fixed
  prefix on the same seeds and steps, 95 % paired bootstrap intervals over
  the audited steps (a missed target counts as an infinite cost), per-seed
  readings and the exact seed-level test with Holm correction.
- Denser audit in the study configurations: every 23rd step (was 47th), at
  every threshold of the curve (0.5–0.99), so that the target accuracy is
  bracketed when decoding for real; each audited step keeps its request
  decisions so that the entropy of the signalling is exact for any subset.
- The report's headline adds the real-decoding comparison; both headline
  tables and the net-rate table compare every receiver with the frozen
  receiver on the same seeds. The published `TABLES.md` compared runs with
  one seed with the three-seed mean, which flattered `nested_two_level`,
  `ewc`, `nested_no_context` and temporal `full_online` by 2–3 points
  (`docs/REVIEW.md`, C8).
- Result files record the code revision at which their process started.
- The signalling-charged teacher-forced figures of a run follow the
  configured first shell (they were read from the shortest one).
- Emulation check table with the verdict of the decision rule
  (`tables/emulation_check_*.md`).

Reproduction
- `--profile published` pins the decoder of the published runs; `core` and
  `paper` use the configured one and repeat the decoder selection as a check.
- `compare_results.py` states the decoder of both folders and does not call
  two folders with different first shells a reproduction.
- The Colab notebook defaults to the confirmatory profile.

## 0.3.1

- Scripts that print a report crashed on Windows consoles, whose default
  code page cannot show the minus sign and the plus-minus sign the tables
  use (`compare_results.py` at the end of a reproduction, found by the test
  suite on Windows 11 with Python 3.14). Every script now forces UTF-8
  output; the test suite checks the comparison report under a legacy code
  page. No measurement is affected.

## 0.3.0 — review release

A strict review of version 0.2 (`docs/REVIEW.md`) found measurement and
reporting gaps. This release closes the ones that can be closed in code and
fixes the write-up. It changes **no** number of the published result set:
every addition is a new measurement next to the existing ones (verified by
test and on the real model).

**Finding.** Six published runs were repeated with this release: they
reproduce the published files bit for bit, and their new measurements show
that the headline saving of version 0.2 was mostly an artefact of emulation
and uncharged signalling (`results/gpt2/RESULTS.md`, section 11). The
refinement decoder has to be improved before the confirmatory runs
(`docs/REVIEW.md`, A7 and F).

Measurement
- Selective refinement now carries its return-link signalling: request bits
  per token as an explicit bitmap and at their entropy, rounds per message,
  and symbols per token at the target accuracy with the signalling charged
  (`refine_grid`, `symbols_at_target_selective_fb*`).
- The same policy is scored for every choice of the first shell.
- Free-running audit on the streams: every `stream.live_every` steps the
  received symbols are decoded for real — selective refinement at the
  thresholds of `stream.live_thresholds` and every fixed prefix — and
  compared with the teacher-forced emulation on the same steps
  (`live_curve`, `live_fixed_curve`).
- Every probe records how stale the receiver's copy is.
- chrF and word error rate next to BLEU; decoded texts are exported
  (`eval/*.texts.json.gz`) and `scripts/add_similarity.py` adds sentence
  similarity from them without re-running.
- Speed benchmark records parameter counts and peak memory; a cost table is
  generated.

Statistics
- Seed-level permutation p-values are printed raw and Holm-corrected,
  together with the smallest p-value the number of seeds can give.
- Equivalence tests (two one-sided tests on per-step differences) with
  margins fixed in the configuration.
- Tables for other target accuracies; tables restricted to a subset of seeds.

Reproduction
- `scripts/reproduce.py --profile {published,core,paper}`: the published
  runs, the six-seed confirmatory protocol, and the sensitivity studies.
- `scripts/compare_results.py` reports provenance, flags mixed code
  revisions, checks the conclusions on all seeds, and lists the new
  measurements.
- Release archive with checksums (`scripts/make_release.py`,
  `scripts/verify_package.py`), VS Code tasks, version lock file, Colab
  notebook.
- A checkpoint serves runs with more adapter slots or another adapter rank.

Fixes
- `TABLES.md` printed one table twice.
- The crossover traffic of a variant run was computed against a context-only
  run that had carried different messages.
- The code revision of a result is recorded in an unpacked archive as well.
- The topic label logged on the temporal stream broke ties through Python's
  string hashing and so differed between runs; it is deterministic now. It
  is a display label: no number depends on it.

## 0.2.0

Token-level redesign (LLM-prior decoding, context memory, LoRA slots) and the
first complete result set with GPT-2 (`results/gpt2`).

## 0.1.0

Sentence-level soft-prompt codec; could not be trained on a CPU
(`docs/METHOD.md`, section 8).
