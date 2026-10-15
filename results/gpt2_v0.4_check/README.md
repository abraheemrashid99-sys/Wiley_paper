# First look at the decoder of version 0.4 on a test stream (seed 0)

Three stream runs of the hierarchical stream, seed 0 — `frozen`,
`context_only` and `nested` — with the decoder chosen on validation text
(`results/decoder_selection`: beam search of width 4, a first round of 6
symbols per token, no reopening) and the denser audit of version 0.4 (every
23rd step, thresholds 0.5–0.99: 26 audited steps, 156 messages per run).
Published codebook, `configs/gpt2.yaml`, one CPU thread per run:

```
python scripts/run_stream.py --config configs/gpt2.yaml --methods frozen --seeds 0 \
    --ckpt <published codebook> out_dir=runs/gpt2_v04_check threads=1 stream.kind=hier
```

(and the same for the other two), then `aggregate.py`, `make_figures.py`,
`export_results.py` and `write_report.py`. Code revisions: `frozen` and
`nested` ran at `2183150`, `context_only` at `9d8bb68`; between the two only
the tables, the tests and the documentation changed
(`git diff --stat 2183150 9d8bb68 -- dnsc scripts/run_stream.py configs` is
empty). Run times: 59, 66 and 87 minutes.

This is a first look, not a result of the paper: one seed, and seed 0 has
been seen before (`docs/METHOD.md`, section 9). The paper's numbers come from
the confirmatory runs (`--profile core`). The discussion is in
`results/gpt2/RESULTS.md`, section 13.

| File | Content |
| --- | --- |
| `TABLES.md` | the report, generated |
| `tables/realdecode_hier.md` | the primary endpoint decoded for real, with paired bootstrap intervals over the audited steps |
| `tables/emulation_check_hier.md` | the emulation check of `docs/METHOD.md`, section 9, with its verdict |
| `stream/*.json` | the exported runs; `raw_streams.tar.gz` holds the full per-step logs |

Two earlier runs of `frozen` and `nested` were made with a version whose
summary read the signalling-charged teacher-forced figures from the policy
starting at the shortest prefix (fixed in `2183150`); they were repeated,
and the repeats are identical to them in every logged value except those two
summary fields and the run time.
