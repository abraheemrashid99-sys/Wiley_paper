**Emulation check on the 'hier' stream (docs/METHOD.md, section 9): free-running against teacher-forced forward symbols per token for 90% token accuracy on exactly the audited steps, curves pooled over the seeds. Verdict: every method within 5% - the teacher-forced curves stand as the measurement, the free-running ones are reported next to them**

| method | seeds | free-running, forward | emulated on the same steps | gap | within 5% |
| --- | --- | --- | --- | --- | --- |
| context_only | 4 | 6.938 | 6.710 | +3.4% | yes |
| frozen | 6 | 7.548 | 7.191 | +5.0% | yes |
| nested | 6 | 6.888 | 6.650 | +3.6% | yes |
