**Selective refinement on europarl_nltk, free-running multi-round decoding**

| channel | snr_db | threshold | sym/token | token acc | BLEU-4 | requests/token | feedback bits/token (bitmap / entropy) | sym/token + feedback (entropy) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| awgn | 10.0 | 0.500 | 5.26 | 0.844 | 0.690 | 1.610 | 4.81 / 3.35 | 7.19 |
| awgn | 10.0 | 0.700 | 6.00 | 0.956 | 0.882 | 1.949 | 4.83 / 3.02 | 7.74 |
| awgn | 10.0 | 0.900 | 6.61 | 0.992 | 0.967 | 2.230 | 4.80 / 2.59 | 8.11 |
| rayleigh | 10.0 | 0.500 | 6.26 | 0.779 | 0.583 | 2.010 | 4.91 / 4.09 | 8.62 |
| rayleigh | 10.0 | 0.700 | 7.09 | 0.935 | 0.810 | 2.346 | 4.84 / 3.80 | 9.29 |
| rayleigh | 10.0 | 0.900 | 7.94 | 0.976 | 0.939 | 2.667 | 4.95 / 3.41 | 9.91 |
