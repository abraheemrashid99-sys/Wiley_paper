# Reproduction check

Reference: `results/gpt2`  ·  reproduction: `results_repro/gpt2_published`  ·  tolerance on headline numbers: 3%

## Verdict

**REPRODUCED: every headline number is within the tolerance and every conclusion holds.**

## Provenance

| Folder | Code revision | torch | transformers | device | CPU threads | Stream runs |
| --- | --- | --- | --- | --- | --- | --- |
| reference | 76d782d | 2.14.1+cu130 | 5.19.0 | cpu | 1 | 11 |
| reference | c5455fa | 2.14.1+cu130 | 5.19.0 | cpu | 1 | 9 |
| reference | ce4a6f4 | 2.14.1+cu130 | 5.19.0 | cpu | 1 | 8 |
| reference | 203a4fa | 2.14.1+cu130 | 5.19.0 | cpu | 1 | 6 |
| reference | 6b803f7 | 2.14.1+cu130 | 5.19.0 | cpu | 1 | 3 |
| reproduction | ed2e023 | 2.11.0+cu130 | 5.19.0 | cuda (Tesla T4) | 1 | 37 |

The reference mixes 5 code revisions (203a4fa, 6b803f7, 76d782d, c5455fa, ce4a6f4).

## Headline numbers (mean over the seeds present in both folders)

| Stream | Method | Seeds | Prior bits/token: published | reproduced | diff | Symbols/token at target: published | reproduced | diff | Within tolerance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hier | context_only | 3 | 5.627 | 5.628 | +0.002 | 5.323 | 5.286 | -0.037 | yes |
| hier | ewc | 1 | 5.412 | 5.414 | +0.001 | 4.923 | 4.874 | -0.049 | yes |
| hier | frozen | 3 | 6.793 | 6.793 | +0.000 | 6.315 | 6.276 | -0.039 | yes |
| hier | full_online | 3 | 5.251 | 5.250 | -0.001 | 4.977 | 4.950 | -0.027 | yes |
| hier | nested | 3 | 5.256 | 5.256 | -0.000 | 4.997 | 4.956 | -0.041 | yes |
| hier | nested_no_context | 1 | 6.431 | 6.435 | +0.005 | 5.717 | 5.723 | +0.006 | yes |
| hier | nested_no_slots | 3 | 5.349 | 5.352 | +0.003 | 5.063 | 5.026 | -0.037 | yes |
| hier | nested_two_level | 1 | 5.302 | 5.312 | +0.010 | 4.842 | 4.794 | -0.048 | yes |
| hier | single_rate | 3 | 5.362 | 5.363 | +0.001 | 5.077 | 5.033 | -0.044 | yes |
| temporal | context_only | 2 | 6.119 | 6.119 | +0.000 | 5.827 | 5.720 | -0.107 | yes |
| temporal | frozen | 2 | 6.331 | 6.331 | +0.000 | 6.007 | 5.957 | -0.050 | yes |
| temporal | full_online | 1 | 5.845 | 5.844 | -0.001 | 5.511 | 5.438 | -0.073 | yes |
| temporal | nested | 2 | 5.852 | 5.857 | +0.005 | 5.642 | 5.572 | -0.070 | yes |
| temporal | single_rate | 2 | 5.857 | 5.861 | +0.005 | 5.651 | 5.580 | -0.071 | yes |

Largest relative difference of a headline number: 1.84%.

## Conclusions of the paper

"Reproduced" uses exactly the runs the reference has.

| Conclusion | Published | Reproduced | Holds in the reproduction |
| --- | --- | --- | --- |
| Selective refinement alone saves at least 20 % of the symbols of a fixed-rate receiver (hierarchical stream, frozen prior) | -33.9% | -33.9% | yes |
| The full system (nested) saves at least 40 % of the symbols of a fixed-rate frozen receiver (hierarchical stream) | -47.7% | -47.8% | yes |
| Prior bits fall in the order frozen > context_only > single_rate > nested (hierarchical stream) | 6.793 > 5.627 > 5.362 > 5.256 | 6.793 > 5.628 > 5.363 > 5.256 | yes |
| Symbols at the target accuracy fall in the order frozen > context_only > single_rate > nested (hierarchical stream) | 6.315 > 5.323 > 5.077 > 4.997 | 6.276 > 5.286 > 5.033 > 4.956 | yes |
| Nested beats the single-rate baseline at a matched synchronisation budget (hierarchical stream; single_rate − nested > 0) | +0.106 bits | +0.107 bits | yes |
| The slots are what helps: nested_no_slots − nested > 0 (hierarchical stream) | +0.093 bits | +0.096 bits | yes |
| Nested is within 0.1 bit of the full-synchronisation upper reference (hierarchical stream) | +0.005 bits | +0.006 bits | yes |
| The upper reference needs more than 20 times the synchronisation traffic of nested | 49x | 49x | yes |
| Prior bits fall in the order frozen > context_only > nested (temporal stream) | 6.331 > 6.119 > 5.852 | 6.331 > 6.119 > 5.857 | yes |
| On the temporal stream nested and single-rate are tied (within 0.1 bit) | +0.005 bits | +0.005 bits | yes |
| The nested codebook loses nothing against a fixed-rate codebook (accuracy difference below 0.02) | 0.8885 nested vs 0.8890 fixed-rate | 0.8915 nested vs 0.8960 fixed-rate | yes |
| BLEU-4 at AWGN 10 dB improves with every part of the prior: llm_ctx > llm > unigram > none | 0.782 > 0.706 > 0.426 > 0.381 | 0.781 > 0.681 > 0.400 > 0.346 | yes |

## Measured by the reproduction only (no published value to compare with)

Real symbols per token for the target token accuracy. "Feedback": the return-link signalling of the refinement requests, charged at the capacity of the mean SNR. "Free-running": the same received symbols decoded for real on a subset of the steps; "emulated": the teacher-forced figure on exactly those steps.

| Stream | Method | Seeds | Fixed prefix | Selective, forward only | + feedback (bitmap) | + feedback (entropy) | Free-running fixed prefix | Free-running selective, forward only | Emulated on the same steps | Free-running + feedback (entropy) | Free-running tok-acc | Emulated tok-acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hier | context_only | 3 | 8.162 | 5.286 | 6.705 | 6.608 | 10.551 | 7.619 | 5.738 | 9.030 | 0.9168 | 0.9561 |
| hier | ewc | 1 | 7.642 | 4.874 | 6.200 | 6.138 | 7.577 | 5.995 | 4.700 | 7.302 | 0.9526 | 0.9660 |
| hier | frozen | 3 | 9.490 | 6.276 | 7.921 | 7.690 | 11.288 | n/a | 6.373 | n/a | 0.9159 | 0.9558 |
| hier | full_online | 3 | 7.771 | 4.950 | 6.292 | 6.226 | 10.016 | n/a | 5.289 | n/a | 0.9304 | 0.9616 |
| hier | nested | 3 | 7.774 | 4.956 | 6.299 | 6.232 | 10.038 | 7.099 | 5.312 | 8.514 | 0.9257 | 0.9622 |
| hier | nested_no_context | 1 | 8.506 | 5.723 | 7.258 | 7.095 | 7.811 | 6.546 | 5.063 | 7.819 | 0.9547 | 0.9755 |
| hier | nested_no_slots | 3 | 7.851 | 5.026 | 6.385 | 6.312 | 10.341 | n/a | 5.426 | n/a | 0.9174 | 0.9586 |
| hier | nested_two_level | 1 | 7.559 | 4.794 | 6.102 | 6.046 | 7.566 | 5.977 | 4.530 | 7.288 | 0.9420 | 0.9679 |
| hier | single_rate | 3 | 7.859 | 5.033 | 6.393 | 6.320 | 10.424 | 7.266 | 5.481 | 8.684 | 0.9243 | 0.9571 |
| temporal | context_only | 2 | 8.543 | 5.720 | 7.239 | 7.095 | 9.929 | n/a | 5.473 | n/a | 0.9142 | 0.9702 |
| temporal | frozen | 2 | 9.004 | 5.957 | 7.534 | 7.355 | 10.225 | n/a | 5.879 | n/a | 0.9160 | 0.9683 |
| temporal | full_online | 1 | 7.902 | 5.438 | 6.900 | 6.784 | 7.524 | 6.088 | 4.564 | 7.361 | 0.9529 | 0.9762 |
| temporal | nested | 2 | 8.246 | 5.572 | 7.060 | 6.932 | 9.830 | 7.278 | 5.299 | 8.636 | 0.9329 | 0.9720 |
| temporal | single_rate | 2 | 8.246 | 5.580 | 7.069 | 6.942 | 9.824 | 7.351 | 5.295 | 8.704 | 0.9287 | 0.9720 |

## Every stream run

| Run | Prior bits: published | reproduced | Symbols at target: published | reproduced | Token accuracy: published | reproduced | Sync kbit/step: published | reproduced |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hier_context_only_s0 | 5.655 | 5.654 | 5.114 | 5.052 | 0.8181 | 0.8229 | 0.00 | 0.00 |
| hier_context_only_s0_rayleigh | 5.832 | 5.839 | 8.617 | 8.568 | 0.7234 | 0.7272 | 0.00 | 0.00 |
| hier_context_only_s1 | 5.613 | 5.610 | 5.364 | 5.296 | 0.8015 | 0.8049 | 0.00 | 0.00 |
| hier_context_only_s2 | 5.611 | 5.622 | 5.492 | 5.509 | 0.7966 | 0.7997 | 0.00 | 0.00 |
| hier_ewc_s0 | 5.412 | 5.414 | 4.923 | 4.874 | 0.8273 | 0.8331 | 23.73 | 23.73 |
| hier_frozen_s0 | 6.843 | 6.843 | 6.038 | 6.031 | 0.7630 | 0.7685 | 0.00 | 0.00 |
| hier_frozen_s0_rayleigh | 6.843 | 6.843 | n/a | n/a | 0.6713 | 0.6736 | 0.00 | 0.00 |
| hier_frozen_s1 | 6.760 | 6.760 | 6.307 | 6.252 | 0.7502 | 0.7537 | 0.00 | 0.00 |
| hier_frozen_s2 | 6.777 | 6.777 | 6.601 | 6.545 | 0.7276 | 0.7359 | 0.00 | 0.00 |
| hier_full_online_s0 | 5.283 | 5.283 | 4.828 | 4.773 | 0.8336 | 0.8396 | 1185.80 | 1185.80 |
| hier_full_online_s1 | 5.254 | 5.245 | 5.023 | 4.982 | 0.8200 | 0.8222 | 1185.80 | 1185.80 |
| hier_full_online_s2 | 5.216 | 5.222 | 5.080 | 5.094 | 0.8173 | 0.8210 | 1185.80 | 1185.80 |
| hier_nested_no_context_s0 | 6.431 | 6.435 | 5.717 | 5.723 | 0.7850 | 0.7904 | 24.23 | 24.23 |
| hier_nested_no_slots_s0 | 5.379 | 5.380 | 4.896 | 4.843 | 0.8288 | 0.8348 | 24.23 | 24.23 |
| hier_nested_no_slots_s1 | 5.340 | 5.335 | 5.094 | 5.042 | 0.8157 | 0.8174 | 24.23 | 24.23 |
| hier_nested_no_slots_s2 | 5.329 | 5.342 | 5.197 | 5.192 | 0.8118 | 0.8151 | 24.23 | 24.23 |
| hier_nested_s0 | 5.284 | 5.291 | 4.829 | 4.782 | 0.8330 | 0.8394 | 24.23 | 24.23 |
| hier_nested_s0_biasonly | 5.546 | 5.550 | 5.031 | 4.970 | 0.8243 | 0.8292 | 0.63 | 0.63 |
| hier_nested_s0_bits4 | 5.284 | 5.291 | 4.830 | 4.782 | 0.8330 | 0.8392 | 12.12 | 12.12 |
| hier_nested_s0_lean | 5.418 | 5.427 | 4.941 | 4.886 | 0.8277 | 0.8331 | 3.31 | 3.31 |
| hier_nested_s0_lr3x | 5.283 | 5.285 | 4.812 | 4.767 | 0.8327 | 0.8390 | 24.23 | 24.23 |
| hier_nested_s0_rayleigh | 5.427 | 5.425 | 7.874 | 7.879 | 0.7444 | 0.7496 | 24.23 | 24.23 |
| hier_nested_s1 | 5.244 | 5.232 | 5.035 | 4.974 | 0.8202 | 0.8221 | 24.23 | 24.23 |
| hier_nested_s2 | 5.240 | 5.245 | 5.127 | 5.111 | 0.8159 | 0.8201 | 24.23 | 24.23 |
| hier_nested_two_level_s0 | 5.302 | 5.312 | 4.842 | 4.794 | 0.8323 | 0.8385 | 29.63 | 29.63 |
| hier_single_rate_s0 | 5.392 | 5.390 | 4.911 | 4.852 | 0.8277 | 0.8339 | 23.73 | 23.73 |
| hier_single_rate_s1 | 5.358 | 5.346 | 5.110 | 5.054 | 0.8138 | 0.8168 | 23.73 | 23.73 |
| hier_single_rate_s2 | 5.336 | 5.352 | 5.208 | 5.191 | 0.8112 | 0.8148 | 23.73 | 23.73 |
| temporal_context_only_s0 | 6.118 | 6.113 | 5.679 | 5.576 | 0.7947 | 0.8037 | 0.00 | 0.00 |
| temporal_context_only_s1 | 6.120 | 6.125 | 5.975 | 5.864 | 0.7823 | 0.7887 | 0.00 | 0.00 |
| temporal_frozen_s0 | 6.334 | 6.334 | 5.835 | 5.797 | 0.7782 | 0.7856 | 0.00 | 0.00 |
| temporal_frozen_s1 | 6.329 | 6.329 | 6.180 | 6.117 | 0.7639 | 0.7702 | 0.00 | 0.00 |
| temporal_full_online_s0 | 5.845 | 5.844 | 5.511 | 5.438 | 0.8071 | 0.8145 | 1185.80 | 1185.80 |
| temporal_nested_s0 | 5.859 | 5.858 | 5.514 | 5.442 | 0.8067 | 0.8138 | 24.23 | 24.23 |
| temporal_nested_s1 | 5.845 | 5.855 | 5.770 | 5.701 | 0.7946 | 0.7996 | 24.23 | 24.23 |
| temporal_single_rate_s0 | 5.864 | 5.865 | 5.520 | 5.452 | 0.8057 | 0.8132 | 23.73 | 23.73 |
| temporal_single_rate_s1 | 5.849 | 5.858 | 5.781 | 5.708 | 0.7944 | 0.7993 | 23.73 | 23.73 |

## Evaluation files and codebooks (all numbers they contain)

| File | Numbers compared | Median abs. difference | Largest abs. difference | Where |
| --- | --- | --- | --- | --- |
| base/pretrain_summary.json | 29 | 0.0010 | 0.0105 | `/history[500]/2` |
| eval/headroom.json | 188 | 0.0000 | 0.1335 | `/rows[ag_news,frozen,ag_news]/ce_bits` |
| eval/rate_europarl_nltk.json | 2566 | 0.0000 | 4.1356 | `/rows[dnsc_llm_ctx,llm_ctx,awgn,0.0,16]/sacrebleu` |
| eval/refine_europarl_nltk.json | 312 | 0.0002 | 5.2944 | `/results[rayleigh,10.0]/free_running[0.5]/sacrebleu` |
| fixed8/pretrain_summary.json | 9 | 0.0000 | 0.0090 | `/history[1500]/8` |

## Generated tables (every number in every table)

| Table | Numbers | Identical to 3 decimals | Median abs. difference | Largest abs. difference |
| --- | --- | --- | --- | --- |
| curves_hier | 144 | 40 | 0.0031 | 0.0700 |
| curves_temporal | 80 | 17 | 0.0035 | 0.1000 |
| headroom | 212 | 47 | 0.0070 | 0.1330 |
| netrate_hier | 94 | 1 | 0.7000 | 3346.0000 |
| netrate_temporal | 51 | 1 | 1.1000 | 16980.0000 |
| rate_europarl_nltk_bleu4 | 192 | 90 | 0.0009 | 0.0460 |
| rate_europarl_nltk_gen_tok_acc | 192 | 79 | 0.0014 | 0.0384 |
| rate_europarl_nltk_power | 12 | 6 | 0.0048 | 0.0150 |
| rate_europarl_nltk_separation | 192 | 192 | 0.0000 | 0.0000 |
| rate_europarl_nltk_tok_acc | 192 | 79 | 0.0014 | 0.0213 |
| refine_europarl_nltk | 72 vs 90 (different layout, e.g. a target reached in only one of them) | | | |
| refine_europarl_nltk_live | 36 vs 54 (different layout, e.g. a target reached in only one of them) | | | |
| regimes_hier | 198 vs 212 (different layout, e.g. a target reached in only one of them) | | | |
| regimes_temporal | 240 vs 252 (different layout, e.g. a target reached in only one of them) | | | |
| sensitivity | 93 vs 94 (different layout, e.g. a target reached in only one of them) | | | |
| stream_hier | 271 vs 281 (different layout, e.g. a target reached in only one of them) | | | |
| stream_temporal | 147 vs 153 (different layout, e.g. a target reached in only one of them) | | | |
