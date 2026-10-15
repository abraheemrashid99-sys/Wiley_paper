# Independent reproduction of `results/gpt2` on a GPU (Colab, Tesla T4)

All 37 published stream runs, the evaluations on the base corpus, the
codebook (trained again from scratch) and the fixed-rate codebook check,
re-run with the code of version 0.3.1 (revision `ed2e023`) on a Google Colab
Tesla T4 with `notebooks/reproduce_gpt2_colab.ipynb`, profile `published`.
Run by the authors on 14–15 October 2026; files unchanged from the archive
the notebook produced (the trained codebook `base/codec.pt` is left out to
keep the folder small; its training log and summary are here).

**Verdict of `COMPARISON.md`: REPRODUCED.** Every headline number is within
the 3 % tolerance of the published value (largest relative difference
1.84 %), and all twelve conclusions checked by `scripts/compare_results.py`
hold. Language-model quantities agree to three decimals; symbols per token
are about 1 % lower on the GPU, the expected size of floating-point
differences between a CPU and a GPU carried through online adaptation.

The section "Measured by the reproduction only" holds the measurements that
version 0.2 did not make: the return-link signalling of selective refinement
and the free-running audit, on three seeds (two on the temporal stream).
They are discussed in `results/gpt2/RESULTS.md`, section 12.
