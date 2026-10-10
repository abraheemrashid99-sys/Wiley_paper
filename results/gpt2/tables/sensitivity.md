**Variant runs (one seed each) next to their reference run on the same stream, messages and noise**

| stream | method | seed | variant | changed | prior bits/token | tok-acc @m_op | selective sym/token | selective tok-acc | sym/token for 90% | sync kbit/step | N* (vs context_only) | slot switches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hier | nested | 0 | (reference) | - | 5.2838 | 0.8330 | 6.028 | 0.9658 | 4.829 | 24.23 | 2735 | 5 |
| hier | nested | 0 | biasonly | nested.lrs.slow=0.0 | 5.5457 | 0.8243 | 6.252 | 0.9659 | 5.031 | 0.63 | 244 | 0 |
| hier | nested | 0 | bits4 | nested.sync.bits=4 | 5.2837 | 0.8330 | 6.028 | 0.9659 | 4.830 | 12.12 | 1374 | 5 |
| hier | nested | 0 | lean | nested.periods.slow=100; nested.sync.bits=4; nested.sync.topk=0.1 | 5.4184 | 0.8277 | 6.167 | 0.9671 | 4.941 | 3.31 | 614 | 2 |
| hier | nested | 0 | lr3x | nested.lrs.medium=0.002; nested.lrs.slow=0.0015 | 5.2832 | 0.8327 | 5.993 | 0.9651 | 4.812 | 24.23 | 2577 | 3 |
