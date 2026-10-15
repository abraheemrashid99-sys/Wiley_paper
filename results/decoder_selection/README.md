# Decoder of selective refinement, chosen on validation text

Produced by

```
python scripts/tune_decoder.py --config configs/gpt2.yaml out_dir=runs/gpt2_verify threads=2
```

with the published GPT-2 codebook, on two CPU cores (stage 1 about 1 h 50
min, stage 2 about 45 min). The candidates, the validation set and the
selection rule were written into `docs/METHOD.md`, section 2.5, before the
script was first run; the one correction made before any candidate was
evaluated (one SNR per message from the stationary law of the SNR process)
is recorded there too. No test material and no stream was used.

| File | Content |
| --- | --- |
| `decoder_selection.json` | every operating point of every candidate (four confidence thresholds), the fixed-prefix curves for beam widths 1, 2 and 4, the decision, the configuration and the environment |
| `decoder_selection.md` / `.csv` / `.tex` | the table generated from it by `scripts/aggregate.py` |
| `superseded_local_beam/` | the first selection, whose beam search ranked sentences by summed per-position posteriors (see below) |

**Decision: beam search of width 4, a first round of 6 symbols per token,
no reopening.** Real symbols per token at 90 % token accuracy, decoded
free-running, with the request signalling charged at its entropy: 8.306,
against 8.971 for the decoder of version 0.3 (greedy, first round of 2
symbols), 9.049 for the fixed prefix decoded with the same beam and 10.994
for the greedy fixed prefix. The discussion, and the limits of the
selection, are in `docs/METHOD.md`, section 2.5 ("Selected decoder").

**Correction.** The first selection used a beam search that ranked partial
sentences by the sum of their per-position posteriors rather than by their
joint posterior. This was found on code review after that selection had
finished and corrected in revision `5f055e3`. Stage 1 compares greedy
decoders only and is unaffected; it was kept. Stage 2 (beam widths 2 and 4
for the stage-1 winner, and the fixed-prefix curves for those widths) was
run again with the corrected beam and the unchanged rule. The decision is
the same as before; the numbers of stage 2 changed (the superseded file
gives 8.248 for the selected candidate and 10.541 for its fixed prefix).

Provenance. `env` is the environment of stage 1, launched at revision
`5305d3f`. The file of that run recorded `9f66973+dirty`, the revision of
the working copy when it was written at the end (the superseded file keeps
that value); the code the selection uses did not change between the two
revisions (`git diff --stat 5305d3f 9f66973 -- dnsc scripts/tune_decoder.py`
lists only an addition to `dnsc/linkbudget.py` that the selection does not
call and the stream runner, which it does not import), so the field was set
by hand to the launch revision before stage 2 was resumed. `resumed` is the
environment of stage 2, launched at `e7baa62`; its `+dirty` mark comes from
the renaming of the superseded result files, which was staged but not yet
committed. From version 0.4 on every result file records the revision at
which its process started.
