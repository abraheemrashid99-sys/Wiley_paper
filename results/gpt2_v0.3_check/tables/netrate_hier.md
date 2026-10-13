**Derived net semantic rate (real symbols per message, payload at equal fidelity plus amortised synchronisation) on the 'hier' stream as a function of messages carried per adaptation step N, at 10 dB**

| method | sym/token for 90% tok-acc | saving vs frozen | payload (sym/msg) | N=1 | N=10 | N=100 | N=1000 | N=10000 | N=100000 | N* (vs frozen) | N* (vs context_only) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| context_only | 5.114 | +15.3% | 92.0 | 92.0 | 92.0 | 92.0 | 92.0 | 92.0 | 92.0 | - | - |
| frozen | 6.038 | +0.0% | 109.0 | 109.0 | 109.0 | 109.0 | 109.0 | 109.0 | 109.0 | - | - |
| nested | 4.829 | +20.0% | 86.9 | 14093.0 | 1487.5 | 226.9 | 100.9 | 88.3 | 87.0 | 644 | 2735 |
| single_rate | 4.911 | +18.7% | 88.4 | 13804.9 | 1460.0 | 225.5 | 102.1 | 89.7 | 88.5 | 677 | 3768 |
