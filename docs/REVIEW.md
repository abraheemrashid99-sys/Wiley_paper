# Review of the version 0.2 study, and what version 0.3 does about it

This is a referee-style review of the code and of the result set in
`results/gpt2`, written before any text of the paper exists. It lists every
finding with the evidence for it, the action taken, and what is still open.
Nothing here changes a number of the published result set; where a new
measurement exists, it is stated where it comes from.

**Verdict on version 0.2: major revision.** The method is coherent, the
pipeline is unusually careful (document-level splits, prequential scoring,
an exact synchronisation ledger, derived quantities kept apart from
measured ones, negative results reported), and the tests are meaningful.
The evidence, however, does not support the claims as they were worded: the
headline rests on a cost that was not charged, on an emulation that was not
audited, and on three seeds. The first audited runs (section E) show that
the first two matter a great deal: decoded for real and with its signalling
charged, the full system saves well under half of what the pilot reported.

Status key — **Fixed**: done and tested in version 0.3. **Fixed in code,
needs the run**: the measurement or protocol exists and is tested, but the
numbers of the paper have to come from the confirmatory runs. **Open**: not
done; what it takes is stated.

## A. Blocking

| # | Finding | Evidence | Action in version 0.3 | Status |
| --- | --- | --- | --- | --- |
| A1 | **The signalling of selective refinement is not charged.** Refinement supplies 71 % of the headline saving, yet after every round the receiver must tell the transmitter which tokens to continue. The fixed-prefix reference needs no such signalling, so the comparison is tilted. | `RESULTS.md`, section 2; `dnsc/refine.py` counted requests but no table charged them. | Every refinement result now carries its request bits per token (explicit bitmap, and the entropy of the decisions), the rounds per message, and symbols per token at the target accuracy with the signalling charged at the capacity of the mean SNR. The same policy is scored for every first shell. | Fixed in code, needs the run. First real values: section E. |
| A2 | **Refinement on the streams is emulated, not decoded.** Each token is judged given the true tokens before it. The headline number is therefore a teacher-forced figure; real decoding with refinement had only been run on 64 base-corpus sentences. | `docs/METHOD.md`, section 7; `dnsc/online.py`. | Free-running audit: every 47th step the same received symbols are decoded for real, round by round, at the thresholds around the target accuracy, and reported next to the emulation on exactly those steps. It draws no random number and moves no other result (tested, and verified on GPT-2). A decision rule fixes when the emulated figure may stand (METHOD, section 9). | Fixed in code, needs the run. First real values: section E. |
| A3 | **Three seeds cannot carry a seed-level claim.** The exact sign-flip test has a floor of 0.25 with three seeds; after Holm correction every comparison in `TABLES.md` reads 1.000. `RESULTS.md` quoted only the block-bootstrap intervals, which pool steps and treat seeds as exchangeable. The ablations, all variants and the Rayleigh stream have one seed; the temporal stream two. | `tables/stream_hier.md`, column "p (Holm)". | Confirmatory protocol with six seeds for the main methods on both streams (floor 0.031) and three for the other ablations; a difference is claimed only if the bootstrap interval excludes zero **and** the seed-level test is significant. Tables print the raw and corrected p-values and the floor. | Fixed in code, needs the run. |
| A4 | **"Indistinguishable" was read off a non-significant test.** The interval of `nested` against `full_online` contains zero, which is absence of evidence. | `RESULTS.md`, section 5. | Two one-sided equivalence tests on the per-step differences with margins fixed in the configuration before the runs (±0.05 bits/token, ±0.05 symbols/token). | Fixed in code, needs the run. |
| A5 | **The result set is not one experiment.** The 37 stream runs carry five code revisions; the code was being optimised while they were launched, and only two runs were repeated afterwards. | `env.git` in the result files: 11, 9, 8, 6 and 3 runs. | One command runs a profile from scratch on one revision; the comparison report shows provenance and flags a folder that mixes revisions. | Fixed in code, needs the run. Section E reports the 6 runs repeated for this review. |
| A6 | **One language model, from 2019.** For a special issue on LLM-enabled semantic communication, GPT-2 (124M) alone will not pass. | — | `configs/laptop.yaml` (SmolLM2-135M, official corpora) runs with the same command. | **Open**: needs a machine that reaches Hugging Face. The configuration has never been run end to end. |
| A7 | **Real decoding is much weaker than the emulation at short prefixes** (found by the audit of A2). The receiver's own wrong decisions mislead the language prior, so the selective policy, which starts at 2 symbols per token, spends rounds and signalling on decisions taken in a corrupted context. With signalling charged and both sides decoded for real, refinement adds only a few percent to what the memories give (section E). | Section E; `live_fixed_curve` and `live_curve` in the result files. | None yet: this is a finding of the review, and the remedy is method work. Candidates, each to be chosen on validation text only: a later first shell, beam search, reopening closed tokens, a prior weight tied to the SNR. The first-shell table gives the emulated side of the first one. | **Open** — to be done before the confirmatory runs. |

## B. Major

| # | Finding | Evidence | Action in version 0.3 | Status |
| --- | --- | --- | --- | --- |
| B1 | **Time nesting is not isolated.** The paper is titled around nesting in time, but the only run in which nothing but the shipping periods differs (`nested_two_level`) has one seed and shows +0.018 bits/token. The advantage over `single_rate` comes from the slots. | `RESULTS.md`, sections 5 and 9. | `nested_two_level` is part of the six-seed protocol; METHOD, section 9, names it as the test of Claim 4 and says the comparison with `single_rate` decides nothing. | Fixed in code, needs the run. If the pilot repeats, the title and Claim 4 must change. |
| B2 | **Retention is compared at unequal freshness.** End-of-regime probes fall on steps 150, 300, 450, 600, where the nested methods have just synchronised; `single_rate` (period 49) is 3, 6, 9 and 12 steps stale. | `dnsc/online.py`; `matched_period`. | Every probe records the staleness of the receiver's copy and the regime table prints it. The retention claim is reworded to rest on `nested` against `nested_no_slots`, which share the phase. | Fixed. |
| B3 | **A baseline result against the method was left out of the text.** On the only free-running fidelity metric of the streams (BLEU-4 on end-of-stream probes) `single_rate` scores 0.623 and `nested` 0.609; the text listed only the methods below `nested` and the upper reference. The metric rests on 16 sentences per domain. | `tables/stream_hier.md`, "retention BLEU-4". | `RESULTS.md` now lists every adapted receiver and says what the metric can and cannot separate. | Fixed. |
| B4 | **No sensitivity analysis**, and `docs/ANALYSIS.md` cited period runs (`slow25`, `slow100`) that do not exist. | Sensitivity table: `bits4`, `lean`, `biasonly`, `lr3x`, `rayleigh` only. | `--profile paper` adds, relative to the configuration: both periods halved and doubled, a 2-bit quantiser, drift twice as fast and twice as slow, mean SNR ±5 dB, context length, adapter rank. `ANALYSIS.md` corrected. | Fixed in code, needs the run. |
| B5 | **Slots were tested only with as many slots as domains.** Allocation is greedy and never recycled; the interesting case is more domains than slots. | `configs/gpt2.yaml`: two domains, two slots. | Three-domain stream with two and with three slots in `--profile paper`; a checkpoint now serves more slots than it was saved with. | Fixed in code, needs the run. |
| B6 | **No semantic fidelity metric** in a semantic-communication paper, and the decoded texts needed to add one were not kept (contrary to the text). | `results/gpt2/eval/rate_europarl_nltk.json` has no texts. | chrF and word error rate are computed everywhere; decoded texts of every prior are exported; `scripts/add_similarity.py` adds sentence-embedding similarity from them. | Fixed in code. Sentence similarity itself: **open** (needs Hugging Face). |
| B7 | **No cost figures**, although the call asks for efficiency and the receiver evaluates a 50 257-way posterior per token. | — | The speed benchmark is part of every profile and records step rates, per-message decode latency at every prefix, parameter counts and peak memory; a cost table is generated. | Fixed in code, needs the run. Energy: **open**. |
| B8 | **No trained end-to-end baseline**, and the ideal-separation reference beats the scheme at a fixed rate (96.5 % of messages delivered at 6 symbols/token against 85.7 % token accuracy). | `RESULTS.md`, section 3. | None in code. The paper must state plainly that the fixed-rate token scheme does not beat ideal separation and that its case rests on graceful degradation and refinement. | **Open**: a DeepSC-style baseline at matched symbols per token. |
| B9 | **The LLM prior hurts at 0 dB** with greedy free-running decoding (token accuracy 0.119 against 0.196 for the unigram receiver at 8 symbols). | `RESULTS.md`, section 9. | None in code; the `snr_low` study (mean 5 dB) will show how far it reaches into the operating range. | **Open**: beam search, or a prior weight tied to the SNR and tuned on validation text only. |
| B10 | **Missing artefacts.** The Colab notebook the README linked to and the codebook checkpoint `RESULTS.md` pointed to were not in the repository. | `notebooks/` absent; `*.pt` ignored. | Notebook added; the codebook was retrained with the published code and reproduces the published training log line by line (45 of 45), so the file added is the published codebook. | Fixed. The notebook has not been executed on a GPU. |

## C. Minor

| # | Finding | Action | Status |
| --- | --- | --- | --- |
| C1 | `TABLES.md` printed the free-running accuracy table twice (overlapping file patterns in `write_report.py`). | Each table is printed once; tested. | Fixed |
| C2 | Calibration error quoted as 0.003–0.006 in one place and 0.005–0.007 in another; the files give 0.002–0.009 over single runs and 0.003–0.006 as per-method means. | Text corrected. | Fixed |
| C3 | The crossover traffic of a variant run was computed against the untagged context-only run whenever the channel type matched, even if the variant carried other messages. No published number was affected (the only such variant, Rayleigh, differed in channel type), but the new sensitivity studies would have been. | Reference must have carried the same messages over the same channel; a context-only run of the same variant is preferred. | Fixed |
| C4 | Results produced from an unpacked archive had no code revision. | `REVISION` file written by the release script; uncommitted changes are marked `+dirty`. | Fixed |
| C5 | Synchronisation is charged at capacity and assumed error-free; the sentence length is assumed known; perfect channel state. | Stated as limitations; the same assumptions now apply to the refinement requests and are stated there too. | Stated |
| C6 | The temporal corpus is a third-party mirror lacking nine months of 2012, and the base corpus is a 14 006-sentence sample of Europarl. | Stated; `configs/laptop.yaml` uses the official sources. | Open (with A6) |
| C7 | The free-running BLEU of the probes rests on 16 sentences per domain. | Stated; the audit of A2 is now the free-running check of the streams (about 1200 tokens per run). | Stated |

## D. Outside the experiments

- The novelty search behind the "first" claims was a handful of web queries. A
  systematic sweep is a prerequisite for the claim, and each reference in
  `docs/METHOD.md`, section 5, must be checked against its published version.
- The square-root law for the shipping periods is a design argument without
  an empirical check until the period studies of B4 are run.

## E. Verification performed for this release

<!-- check:begin -->
- **Tests.** 86 tests pass (34 unit, 52 integration) on Linux, CPU, PyTorch 2.14.1, Transformers 5.19.0, including the tests added for every new code path.
- **Published codebook.** Retraining with the published code and
  configuration reproduced the published training log line by line (45 of 45
  logged losses and accuracies identical); the resulting file is
  `results/gpt2/base/codec.pt`.
- **Published stream runs.** 6 of 37 were repeated with the corrected
  code on real GPT-2 from one clean revision (`a25f99d`): `hier_context_only_s0`, `hier_frozen_s0`, `hier_nested_s0`, `hier_single_rate_s0`, `temporal_frozen_s0`, `temporal_nested_s0`. All
  6 are identical to the published per-step logs in every measured
  value (116,684 stored values compared; the only 254 that
  differ are topic labels of the temporal stream, a display label with a
  hash-dependent tie-break, now deterministic). This also shows on the real
  model that the new measurements move nothing else. The files and the
  generated comparison report are in `results/gpt2_v0.3_check`.
- **Not verified.** The other 31 published runs and the base-corpus
  evaluations were not repeated; no GPU, no Windows or macOS machine and no
  second backbone was used; the Colab notebook was not executed.

### First values of the new measurements (seed 0, hierarchical stream)

These come from the repeated runs above and are a first look, not a result.
Symbols per token for 90 % token accuracy; full tables in
`results/gpt2/RESULTS.md`, section 11.

| Full system (`nested`, selective refinement) against the frozen fixed-prefix receiver | Saving |
| --- | --- |
| As published: refinement emulated teacher-forced, signalling not charged | 46.9 % |
| Emulated, signalling charged at its entropy (as a bitmap) | 33.0 % (32.4 %) |
| Both sides decoded for real on the audited steps, signalling charged at its entropy (as a bitmap) | 17.9 % (14.4 %) |
| of which: the memories alone, at a fixed prefix, decoded for real | 10.5 % |

Decoded for real, selective refinement needs 29.6 % (`nested`) to
33.2 % (`frozen`) more forward symbols than its emulation on the
same steps. On the temporal stream (`results/gpt2/RESULTS.md`, section 11)
the full system saves 7.6 % (2.0 %) where the pilot's measure
gave 37.4 %. The emulation check of METHOD, section 9, therefore fails, and
the free-running figures are the ones to report.

**Consequence for the paper.** On this evidence the large refinement gain of
the pilot was mostly an artefact of emulation and uncharged signalling. What
survives is smaller and of a different composition: the memories give about a
tenth at a fixed prefix with no signalling, and refinement adds less than
that once it is decoded for real and paid for. One seed and 72 messages per
run cannot fix these numbers, but they are enough to say that the pilot's
headline must not be written into a paper.
<!-- check:end -->

## F. What has to happen before the paper is written

**First, the decoder (A7).** The first look in section E says that the
simplest refinement policy does not pay for itself once it is decoded for
real. Before any confirmatory run:

1. Extend the base-corpus refinement evaluation (`scripts/eval_refine.py`)
   to decode for real with a later first shell, with beam search and with
   reopening of closed tokens, on the **validation** split of the base
   corpus only.
2. Choose one decoder and one first shell there, by total symbols (forward
   plus signalling) at the target accuracy, and write the choice into
   `docs/METHOD.md` before a stream is touched.
3. If no variant beats the fixed prefix decoded for real, selective
   refinement leaves the paper's claims, and the paper is about the memories:
   they need no signalling, and their gain held under real decoding.

**Then the confirmatory runs** (`--profile core`), which must show:

1. The saving against the frozen fixed-prefix receiver, **both decoded for
   real and with signalling charged**, on six seeds.
2. Seed-level significance of context against no memory, of slots, and of
   time nesting, each tested by the comparison named in METHOD, section 9.
3. Equivalence of `nested` and `full_online` within the fixed margins.
4. The same on the three seeds the pilot never saw.

Whatever these show is the result, and the paper is framed around it.
