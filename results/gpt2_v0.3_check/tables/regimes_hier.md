**The 'hier' stream by slow regime: prequential metrics while the regime is active, and held-out probes of it at the end of the stream (mean ± SD over seeds). A probe scores the receiver's copy at that moment: methods are comparable on a probe only at equal staleness (steps since the slowest level was shipped)**

| method | regime | seeds | prior bits/token | tok-acc @m_op | selective sym/token | selective tok-acc | held-out prior bits (end) | held-out tok-acc (end) | receiver staleness at the probe (steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| context_only | ag_news | 1 | 6.2006 ± 0.0000 | 0.8166 ± 0.0000 | 6.3795 ± 0.0000 | 0.9664 ± 0.0000 | 6.3420 | 0.8316 | - |
| context_only | teleqna | 1 | 5.1101 ± 0.0000 | 0.8195 ± 0.0000 | 6.2423 ± 0.0000 | 0.9633 ± 0.0000 | 4.8965 | 0.8782 | - |
| frozen | ag_news | 1 | 6.5875 ± 0.0000 | 0.7992 ± 0.0000 | 6.6441 ± 0.0000 | 0.9665 ± 0.0000 | 6.6539 | 0.8081 | - |
| frozen | teleqna | 1 | 7.0986 ± 0.0000 | 0.7269 ± 0.0000 | 7.6597 ± 0.0000 | 0.9590 ± 0.0000 | 7.2270 | 0.7915 | - |
| nested | ag_news | 1 | 5.9723 ± 0.0000 | 0.8244 ± 0.0000 | 6.2732 ± 0.0000 | 0.9687 ± 0.0000 | 5.8482 | 0.8349 | 150 |
| nested | teleqna | 1 | 4.5952 ± 0.0000 | 0.8416 ± 0.0000 | 5.7829 ± 0.0000 | 0.9629 ± 0.0000 | 4.1277 | 0.9091 | 150 |
| single_rate | ag_news | 1 | 6.0578 ± 0.0000 | 0.8197 ± 0.0000 | 6.3148 ± 0.0000 | 0.9674 ± 0.0000 | 6.1996 | 0.8375 | 12 |
| single_rate | teleqna | 1 | 4.7265 ± 0.0000 | 0.8357 ± 0.0000 | 5.9051 ± 0.0000 | 0.9638 ± 0.0000 | 4.1983 | 0.9039 | 12 |
