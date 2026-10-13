# Reproduction check

Reference: `results/gpt2`  ·  reproduction: `results/gpt2_v0.3_check`  ·  tolerance on headline numbers: 3%

## Verdict

**CHECK NEEDED:**

- 31 stream runs of the reference are missing in the reproduction
- conclusion could not be checked: The slots are what helps: nested_no_slots − nested > 0 (hierarchical stream)
- conclusion could not be checked: Nested is within 0.1 bit of the full-synchronisation upper reference (hierarchical stream)
- conclusion could not be checked: The upper reference needs more than 20 times the synchronisation traffic of nested
- conclusion could not be checked: Prior bits fall in the order frozen > context_only > nested (temporal stream)
- conclusion could not be checked: On the temporal stream nested and single-rate are tied (within 0.1 bit)
- conclusion could not be checked: The nested codebook loses nothing against a fixed-rate codebook (accuracy difference below 0.02)
- conclusion could not be checked: BLEU-4 at AWGN 10 dB improves with every part of the prior: llm_ctx > llm > unigram > none
- eval/headroom.json is missing in the reproduction
- eval/rate_europarl_nltk.json is missing in the reproduction
- eval/refine_europarl_nltk.json is missing in the reproduction
- fixed8/pretrain_summary.json is missing in the reproduction

## Provenance

| Folder | Code revision | torch | transformers | device | CPU threads | Stream runs |
| --- | --- | --- | --- | --- | --- | --- |
| reference | 76d782d | 2.14.1+cu130 | 5.19.0 | cpu | 1 | 11 |
| reference | c5455fa | 2.14.1+cu130 | 5.19.0 | cpu | 1 | 9 |
| reference | ce4a6f4 | 2.14.1+cu130 | 5.19.0 | cpu | 1 | 8 |
| reference | 203a4fa | 2.14.1+cu130 | 5.19.0 | cpu | 1 | 6 |
| reference | 6b803f7 | 2.14.1+cu130 | 5.19.0 | cpu | 1 | 3 |
| reproduction | a25f99d | 2.14.1+cu130 | 5.19.0 | cpu | 1 | 6 |

The reference mixes 5 code revisions (203a4fa, 6b803f7, 76d782d, c5455fa, ce4a6f4).

## Headline numbers (mean over the seeds present in both folders)

| Stream | Method | Seeds | Prior bits/token: published | reproduced | diff | Symbols/token at target: published | reproduced | diff | Within tolerance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hier | context_only | 1 | 5.655 | 5.655 | +0.000 | 5.114 | 5.114 | +0.000 | yes |
| hier | frozen | 1 | 6.843 | 6.843 | +0.000 | 6.038 | 6.038 | +0.000 | yes |
| hier | nested | 1 | 5.284 | 5.284 | +0.000 | 4.829 | 4.829 | +0.000 | yes |
| hier | single_rate | 1 | 5.392 | 5.392 | +0.000 | 4.911 | 4.911 | +0.000 | yes |
| temporal | frozen | 1 | 6.334 | 6.334 | +0.000 | 5.835 | 5.835 | +0.000 | yes |
| temporal | nested | 1 | 5.859 | 5.859 | +0.000 | 5.514 | 5.514 | +0.000 | yes |

Largest relative difference of a headline number: 0.00%.

Missing in the reproduction (31): `hier_context_only_s0_rayleigh`, `hier_context_only_s1`, `hier_context_only_s2`, `hier_ewc_s0`, `hier_frozen_s0_rayleigh`, `hier_frozen_s1`, `hier_frozen_s2`, `hier_full_online_s0`, `hier_full_online_s1`, `hier_full_online_s2`, `hier_nested_no_context_s0`, `hier_nested_no_slots_s0`, `hier_nested_no_slots_s1`, `hier_nested_no_slots_s2`, `hier_nested_s0_biasonly`, `hier_nested_s0_bits4`, `hier_nested_s0_lean`, `hier_nested_s0_lr3x`, `hier_nested_s0_rayleigh`, `hier_nested_s1`, `hier_nested_s2`, `hier_nested_two_level_s0`, `hier_single_rate_s1`, `hier_single_rate_s2`, `temporal_context_only_s0`, `temporal_context_only_s1`, `temporal_frozen_s1`, `temporal_full_online_s0`, `temporal_nested_s1`, `temporal_single_rate_s0`, `temporal_single_rate_s1`

## Conclusions of the paper

"Reproduced" uses exactly the runs the reference has.

| Conclusion | Published | Reproduced | Holds in the reproduction |
| --- | --- | --- | --- |
| Selective refinement alone saves at least 20 % of the symbols of a fixed-rate receiver (hierarchical stream, frozen prior) | -33.9% | -33.6% | yes |
| The full system (nested) saves at least 40 % of the symbols of a fixed-rate frozen receiver (hierarchical stream) | -47.7% | -46.9% | yes |
| Prior bits fall in the order frozen > context_only > single_rate > nested (hierarchical stream) | 6.793 > 5.627 > 5.362 > 5.256 | 6.843 > 5.655 > 5.392 > 5.284 | yes |
| Symbols at the target accuracy fall in the order frozen > context_only > single_rate > nested (hierarchical stream) | 6.315 > 5.323 > 5.077 > 4.997 | 6.038 > 5.114 > 4.911 > 4.829 | yes |
| Nested beats the single-rate baseline at a matched synchronisation budget (hierarchical stream; single_rate − nested > 0) | +0.106 bits | +0.108 bits | yes |
| The slots are what helps: nested_no_slots − nested > 0 (hierarchical stream) | +0.093 bits | not available | n/a |
| Nested is within 0.1 bit of the full-synchronisation upper reference (hierarchical stream) | +0.005 bits | not available | n/a |
| The upper reference needs more than 20 times the synchronisation traffic of nested | 49x | not available | n/a |
| Prior bits fall in the order frozen > context_only > nested (temporal stream) | 6.331 > 6.119 > 5.852 | not available | n/a |
| On the temporal stream nested and single-rate are tied (within 0.1 bit) | +0.005 bits | not available | n/a |
| The nested codebook loses nothing against a fixed-rate codebook (accuracy difference below 0.02) | 0.8885 nested vs 0.8890 fixed-rate | not available | n/a |
| BLEU-4 at AWGN 10 dB improves with every part of the prior: llm_ctx > llm > unigram > none | 0.782 > 0.706 > 0.426 > 0.381 | not available | n/a |

## Measured by the reproduction only (no published value to compare with)

Real symbols per token for the target token accuracy. "Feedback": the return-link signalling of the refinement requests, charged at the capacity of the mean SNR. "Free-running": the same received symbols decoded for real on a subset of the steps; "emulated": the teacher-forced figure on exactly those steps.

| Stream | Method | Seeds | Fixed prefix | Selective, forward only | + feedback (bitmap) | + feedback (entropy) | Free-running fixed prefix | Free-running selective, forward only | Emulated on the same steps | Free-running + feedback (entropy) | Free-running tok-acc | Emulated tok-acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hier | context_only | 1 | 7.872 | 5.114 | 6.498 | 6.417 | 7.832 | 6.201 | 4.617 | 7.500 | 0.9558 | 0.9801 |
| hier | frozen | 1 | 9.089 | 6.038 | 7.639 | 7.435 | 8.531 | 7.111 | 5.337 | 8.427 | 0.9373 | 0.9760 |
| hier | nested | 1 | 7.664 | 4.829 | 6.144 | 6.087 | 7.634 | 5.709 | 4.407 | 7.004 | 0.9557 | 0.9766 |
| hier | single_rate | 1 | 7.754 | 4.911 | 6.245 | 6.183 | 7.862 | 5.959 | 4.537 | 7.281 | 0.9499 | 0.9814 |
| temporal | frozen | 1 | 8.813 | 5.835 | 7.390 | 7.224 | 8.180 | 6.908 | 5.039 | 8.207 | 0.9169 | 0.9758 |
| temporal | nested | 1 | 8.044 | 5.514 | 6.993 | 6.869 | 7.630 | 6.277 | 4.611 | 7.556 | 0.9461 | 0.9769 |

## Every stream run

| Run | Prior bits: published | reproduced | Symbols at target: published | reproduced | Token accuracy: published | reproduced | Sync kbit/step: published | reproduced |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hier_context_only_s0 | 5.655 | 5.655 | 5.114 | 5.114 | 0.8181 | 0.8181 | 0.00 | 0.00 |
| hier_frozen_s0 | 6.843 | 6.843 | 6.038 | 6.038 | 0.7630 | 0.7630 | 0.00 | 0.00 |
| hier_nested_s0 | 5.284 | 5.284 | 4.829 | 4.829 | 0.8330 | 0.8330 | 24.23 | 24.23 |
| hier_single_rate_s0 | 5.392 | 5.392 | 4.911 | 4.911 | 0.8277 | 0.8277 | 23.73 | 23.73 |
| temporal_frozen_s0 | 6.334 | 6.334 | 5.835 | 5.835 | 0.7782 | 0.7782 | 0.00 | 0.00 |
| temporal_nested_s0 | 5.859 | 5.859 | 5.514 | 5.514 | 0.8067 | 0.8067 | 24.23 | 24.23 |

## Evaluation files and codebooks (all numbers they contain)

| File | Numbers compared | Median abs. difference | Largest abs. difference | Where |
| --- | --- | --- | --- | --- |
| base/pretrain_summary.json | 29 | 0.0000 | 0.0000 | `/steps` |
| eval/headroom.json | missing in the reproduction | | | |
| eval/rate_europarl_nltk.json | missing in the reproduction | | | |
| eval/refine_europarl_nltk.json | missing in the reproduction | | | |
| fixed8/pretrain_summary.json | missing in the reproduction | | | |

## Generated tables (every number in every table)

| Table | Numbers | Identical to 3 decimals | Median abs. difference | Largest abs. difference |
| --- | --- | --- | --- | --- |
| curves_hier | 144 vs 64 (different layout, e.g. a target reached in only one of them) | | | |
| curves_temporal | 80 vs 32 (different layout, e.g. a target reached in only one of them) | | | |
| headroom | missing in the reproduction | | | |
| netrate_hier | 94 vs 40 (different layout, e.g. a target reached in only one of them) | | | |
| netrate_temporal | 51 vs 19 (different layout, e.g. a target reached in only one of them) | | | |
| rate_europarl_nltk_bleu4 | missing in the reproduction | | | |
| rate_europarl_nltk_gen_tok_acc | missing in the reproduction | | | |
| rate_europarl_nltk_power | missing in the reproduction | | | |
| rate_europarl_nltk_separation | missing in the reproduction | | | |
| rate_europarl_nltk_tok_acc | missing in the reproduction | | | |
| refine_europarl_nltk | missing in the reproduction | | | |
| refine_europarl_nltk_live | missing in the reproduction | | | |
| regimes_hier | 198 vs 92 (different layout, e.g. a target reached in only one of them) | | | |
| regimes_temporal | 240 vs 100 (different layout, e.g. a target reached in only one of them) | | | |
| sensitivity | missing in the reproduction | | | |
| stream_hier | 271 vs 111 (different layout, e.g. a target reached in only one of them) | | | |
| stream_temporal | 147 vs 51 (different layout, e.g. a target reached in only one of them) | | | |
