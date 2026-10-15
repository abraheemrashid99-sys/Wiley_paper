# Decoder of selective refinement, chosen on validation text

Produced by

```
python scripts/tune_decoder.py --config configs/gpt2.yaml out_dir=runs/gpt2_verify threads=2
```

with the published GPT-2 codebook, on two CPU cores (about 2 h 40 min).
The candidates, the validation set and the selection rule were written into
`docs/METHOD.md`, section 2.5, before the script was first run; the one
correction made before any candidate was evaluated (one SNR per message from
the stationary law of the SNR process) is recorded there too. No test
material and no stream was used.

| File | Content |
| --- | --- |
| `decoder_selection.json` | every operating point of every candidate (four confidence thresholds), the fixed-prefix curves for beam widths 1, 2 and 4, the decision, the configuration and the environment |
| `decoder_selection.md` / `.csv` / `.tex` | the table generated from it by `scripts/aggregate.py` |

**Decision: beam search of width 4, a first round of 6 symbols per token,
no reopening.** Real symbols per token at 90 % token accuracy, decoded
free-running, with the request signalling charged at its entropy: 8.248,
against 8.971 for the decoder of version 0.3 (greedy, first round of 2
symbols) and 10.541 for the fixed prefix decoded with the same beam (10.994
greedy). The discussion, and the two limits of the selection, are in
`docs/METHOD.md`, section 2.5 ("Selected decoder").

Provenance: the field `env.git` reads `9f66973+dirty`, the revision of the
working copy when the file was written at the end. The selection was
launched at revision `5305d3f`; between the two, functions the selection
does not call were added to `dnsc/linkbudget.py`, and `dnsc/online.py` (the
stream runner, which the selection does not import) was changed
(`git diff --stat 5305d3f 9f66973 -- dnsc scripts/tune_decoder.py`). Since version
0.4 every result file records the revision at which its process started.
