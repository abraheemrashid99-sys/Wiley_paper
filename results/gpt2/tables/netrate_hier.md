**Derived net semantic rate (real symbols per message, payload at equal fidelity plus amortised synchronisation) on the 'hier' stream as a function of messages carried per adaptation step N, at 10 dB**

| method | sym/token for 90% tok-acc | saving vs frozen | payload (sym/msg) | N=1 | N=10 | N=100 | N=1000 | N=10000 | N=100000 | N* (vs frozen) | N* (vs context_only) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| context_only | 5.323 | +15.7% | 95.5 | 95.5 | 95.5 | 95.5 | 95.5 | 95.5 | 95.5 | - | - |
| frozen | 6.315 | +0.0% | 113.6 | 113.6 | 113.6 | 113.6 | 113.6 | 113.6 | 113.6 | - | - |
| full_online | 4.977 | +21.2% | 89.3 | 685633.8 | 68643.7 | 6944.7 | 774.8 | 157.8 | 96.1 | 28562 | 110411 |
| nested | 4.997 | +20.9% | 89.6 | 14095.7 | 1490.2 | 229.7 | 103.6 | 91.0 | 89.8 | 592 | 2394 |
| nested_no_context | 5.717 | +9.5% | 103.2 | 14109.3 | 1503.8 | 243.3 | 117.2 | 104.6 | 103.3 | 1298 | never |
| nested_no_slots | 4.896 | +22.5% | 88.1 | 14094.0 | 1488.7 | 228.1 | 102.1 | 89.5 | 88.2 | 548 | 1822 |
| single_rate | 5.077 | +19.6% | 91.0 | 13807.6 | 1462.7 | 228.2 | 104.8 | 92.4 | 91.2 | 617 | 3101 |
