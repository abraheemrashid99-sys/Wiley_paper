**Decoder of selective refinement, chosen on validation text only (96 messages, 1633 tokens; SNR mean 9.8 dB): real symbols per token for 90% token accuracy, decoded free-running, the signalling of the requests charged at its entropy (criterion) or as a bitmap; the fixed prefix is decoded with the same beam. Rule fixed in docs/METHOD.md, section 2.5: lowest criterion among the greedy candidates (ties within 1% go to fewer rounds), then a beam only if it lowers the criterion by more than 2%**

| candidate | beam | first shell (symbols) | reopening | forward only | + signalling (entropy) | + signalling (bitmap) | fixed prefix, same beam | vs fixed prefix (entropy) | rounds/message at 0.95 | tok-acc at 0.95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| beam1_start0_closed | 1 | 2 | no | 7.527 | 8.971 | 9.442 | 10.994 | -18.4% | 5.00 | 0.930 |
| beam1_start0_reopen | 1 | 2 | yes | 7.486 | 9.937 | 10.541 | 10.994 | -9.6% | 5.34 | 0.972 |
| beam1_start1_closed | 1 | 4 | no | 7.395 | 8.652 | 8.740 | 10.994 | -21.3% | 4.01 | 0.957 |
| beam1_start1_reopen | 1 | 4 | yes | 7.682 | 9.640 | 10.060 | 10.994 | -12.3% | 4.22 | 0.973 |
| beam1_start2_closed | 1 | 6 | no | 7.836 | 8.730 | 8.758 | 10.994 | -20.6% | 3.01 | 0.966 |
| beam1_start2_reopen | 1 | 6 | yes | 8.211 | 9.533 | 9.948 | 10.994 | -13.3% | 3.20 | 0.974 |
| beam2_start2_closed | 2 | 6 | no | 7.738 | 8.603 | 8.637 | 10.044 | -14.3% | 3.01 | 0.983 |
| beam4_start2_closed (chosen) | 4 | 6 | no | 7.492 | 8.306 | 8.357 | 9.049 | -8.2% | 3.01 | 0.979 |
