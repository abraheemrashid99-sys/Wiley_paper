# Primary comparison with the decoder of version 0.4: six seeds

The hierarchical drift stream with GPT-2 (124M) as the frozen prior, the
published codebook and the decoder chosen on validation text
(`results/decoder_selection`: beam 4, first round of 6 symbols, no
reopening); the denser audit of version 0.4 (every 23rd step, 156 messages
per run). Receivers: `frozen` and `nested` on seeds 0–5, `context_only` on
seeds 0, 3, 4 and 5.

| Seeds | Machine | Code revision of the run | How |
| --- | --- | --- | --- |
| 0 | Linux CPU, 1 thread per run | `2183150` (`context_only`: `9d8bb68`) | `scripts/run_stream.py`, see `results/gpt2_v0.4_check` |
| 1, 2 (`frozen`, `nested`) | Linux CPU, 1 thread per run | `3b0eb68` | `scripts/run_stream.py` |
| 3, 4, 5 | Windows 11 laptop CPU, 2 threads per run, 4 runs in parallel | `ace8bab` (release 0.4.1) | `scripts/run_quick.py` |

The stream runner and the decoder are the same in all these revisions
(`git diff 2183150 ace8bab -- dnsc` changes only the version string). The
`context_only` runs of seeds 1 and 2 were started but stopped when the
machine was restarted; they were not repeated. Seeds 3–5 had never been run
before. The runs of seeds 0–2 in this folder are byte-identical to the ones
made here; the laptop pooled them with its own runs
(`run_quick.py --report-only --add-runs seeds_0_1_2.zip`), and the tables
were generated again here from the same raw files with the same result
(apart from line endings).

| File | Content |
| --- | --- |
| `tables/realdecode_hier.md` | primary endpoint decoded for real, with 95 % paired bootstrap intervals, per-seed readings and the seed-level test |
| `tables/emulation_check_hier.md` | the decision rule of `docs/METHOD.md`, section 9 (all within 5 %: the teacher-forced curves are the measurement) |
| `tables/refine_cost_hier.md`, `stream_hier.md`, `netrate_hier.md`, … | teacher-forced figures, signalling, prior cross-entropy, synchronisation and the crossover traffic |
| `TABLES.md` | the report |
| `paper/results_section.tex` | the results section of the paper, with the abstract and conclusion numbers in its header; every number from these files |
| `paper/fig_primary_endpoint.pdf` | its figure (`scripts/paper_figure_endpoint.py`) |
| `stream/*.json`, `raw_streams.tar.gz` | the exported runs and the full per-step logs |

Headline (all from the tables): DN-SC needs 7.08 real channel symbols per
token for 90 % token accuracy, teacher-forced with the return-link
signalling charged, against 9.23 for the frozen fixed-rate receiver with the
same decoder: −23.3 %, lower on all six seeds (p = 0.031). Decoded for real
on the audited steps: −27.9 % [−31.7 %, −21.6 %]. The discussion is in
`results/gpt2/RESULTS.md`, section 14.
