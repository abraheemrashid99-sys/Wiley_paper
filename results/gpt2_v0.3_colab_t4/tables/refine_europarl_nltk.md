**Selective refinement on europarl_nltk, teacher-forced sweep (token accuracy; ECE of token confidence)**

| channel | snr_db | ECE raw | ECE calibrated | best fixed m | fixed acc | selective sym/token | selective acc | saving | feedback bits/token (bitmap / entropy) | saving, feedback charged (entropy) | max gain vs fixed | oracle sym/token | random control acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| awgn | 0.0 | 0.019 | 0.008 | 16 | 0.786 | 13.51 | 0.782 | 15.6% | 4.57 / 1.57 | -4.1% | +0.103 @ 9.41 | 8.19 | 0.550 |
| awgn | 10.0 | 0.012 | 0.007 | 16 | 1.000 | 6.42 | 0.998 | 59.9% | 3.09 / 2.17 | 52.0% | +0.142 @ 4.38 | 3.76 | 0.898 |
| awgn | 20.0 | 0.003 | 0.006 | 6 | 1.000 | 3.20 | 0.995 | 46.7% | 1.60 / 1.34 | 40.0% | +0.118 @ 2.59 | 2.56 | 0.957 |
| rayleigh | 0.0 | 0.019 | 0.014 | 16 | 0.639 | 14.06 | 0.638 | 12.1% | 4.66 / 1.31 | -4.3% | +0.044 @ 11.61 | 9.12 | 0.474 |
| rayleigh | 10.0 | 0.014 | 0.006 | 16 | 0.942 | 8.88 | 0.940 | 44.5% | 3.61 / 2.52 | 35.4% | +0.096 @ 7.37 | 5.11 | 0.763 |
| rayleigh | 20.0 | 0.007 | 0.007 | 16 | 0.996 | 4.50 | 0.992 | 71.9% | 2.16 / 1.99 | 68.1% | +0.116 @ 3.20 | 3.02 | 0.918 |
