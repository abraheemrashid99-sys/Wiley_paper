**Emulation check on the 'hier' stream (docs/METHOD.md, section 9): free-running against teacher-forced forward symbols per token for 90% token accuracy on exactly the audited steps, curves pooled over the seeds. Verdict: not every method within 5% - the free-running figures are the measurement, on both sides of the comparison**

| method | seeds | free-running, forward | emulated on the same steps | gap | within 5% |
| --- | --- | --- | --- | --- | --- |
| context_only | 3 | 7.610 | 5.727 | +32.9% | NO |
| ewc | 1 | 5.995 | 4.700 | +27.5% | NO |
| frozen | 3 | 8.352 | 6.361 | +31.3% | NO |
| full_online | 3 | 7.062 | 5.290 | +33.5% | NO |
| nested | 3 | 7.097 | 5.307 | +33.7% | NO |
| nested_no_context | 1 | 6.546 | 5.063 | +29.3% | NO |
| nested_no_slots | 3 | 7.255 | 5.419 | +33.9% | NO |
| nested_two_level | 1 | 5.977 | 4.530 | +31.9% | NO |
| single_rate | 3 | 7.264 | 5.470 | +32.8% | NO |
