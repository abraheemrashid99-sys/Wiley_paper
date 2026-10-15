# The Colab runs, re-analysed with the scripts of version 0.4

The runs in the folder above were not changed or repeated. Their per-step
logs (`../raw_streams.tar.gz`) were aggregated again with
`scripts/aggregate.py` and `scripts/write_report.py` at revision `69de7a1`:

```
tar -xzf results/gpt2_v0.3_colab_t4/raw_streams.tar.gz -C <dir>/stream
python scripts/aggregate.py --config configs/gpt2.yaml out_dir=<dir>
python scripts/write_report.py --config configs/gpt2.yaml out_dir=<dir> --to <export>
```

(with the exported files, figures and evaluations of the folder above copied
into `<export>`). Every other table comes out identical to the one in
`../tables`; only these differ or are new:

| File | What changed |
| --- | --- |
| `tables/realdecode_hier.md`, `tables/realdecode_temporal.md` | new: the primary endpoint decoded for real, pooled over the seeds, with 95 % paired bootstrap intervals over the audited steps, per-seed readings and the seed-level test |
| `tables/emulation_check_hier.md`, `tables/emulation_check_temporal.md` | new: the emulation check of `docs/METHOD.md`, section 9 (free-running against teacher-forced forward symbols on the audited steps) and its verdict — here 27–34 % (hierarchical) and 33–40 % (temporal), so the free-running figures are the measurement |
| `tables/refine_cost_*.md` | the free-running columns are read off the curves pooled over the seeds (before: mean of per-run readings, "not reached" whenever one seed's few audited steps missed the target) |
| `tables/netrate_*.md` | the saving and the crossover of a run with one seed are computed against the frozen and the context-only receiver on the same seed (`docs/REVIEW.md`, C8) |
| `TABLES.md` | the report: the headline compares on matched seeds and adds the comparison decoded for real |

These runs used the decoder of version 0.3 (greedy, first round of 2
symbols, no reopening) and audited every 47th step at thresholds 0.5, 0.7
and 0.9. Each audited step of these runs did not yet keep its request
decisions, so in the bootstrap the entropy of the signalling is held at each
run's own value; the bitmap figures are exact. The discussion is in
`results/gpt2/RESULTS.md`, section 12.
