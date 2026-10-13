**The 'temporal' stream by slow regime: prequential metrics while the regime is active, and held-out probes of it at the end of the stream (mean ± SD over seeds). A probe scores the receiver's copy at that moment: methods are comparable on a probe only at equal staleness (steps since the slowest level was shipped)**

| method | regime | seeds | prior bits/token | tok-acc @m_op | selective sym/token | selective tok-acc | held-out prior bits (end) | held-out tok-acc (end) | receiver staleness at the probe (steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen | era0 | 1 | 6.2021 ± 0.0000 | 0.8515 ± 0.0000 | 6.0381 ± 0.0000 | 0.9747 ± 0.0000 | 6.5289 | 0.7936 | - |
| frozen | era1 | 1 | 6.2595 ± 0.0000 | 0.7088 ± 0.0000 | 8.1732 ± 0.0000 | 0.9493 ± 0.0000 | 6.5974 | 0.8025 | - |
| frozen | era2 | 1 | 6.4015 ± 0.0000 | 0.7513 ± 0.0000 | 7.4670 ± 0.0000 | 0.9664 ± 0.0000 | 6.5500 | 0.7815 | - |
| frozen | era3 | 1 | 6.4729 ± 0.0000 | 0.8014 ± 0.0000 | 6.7072 ± 0.0000 | 0.9718 ± 0.0000 | 6.4767 | 0.7944 | - |
| nested | era0 | 1 | 5.7896 ± 0.0000 | 0.8691 ± 0.0000 | 5.6797 ± 0.0000 | 0.9738 ± 0.0000 | 6.0098 | 0.8071 | 600 |
| nested | era1 | 1 | 5.7854 ± 0.0000 | 0.7418 ± 0.0000 | 7.7313 ± 0.0000 | 0.9502 ± 0.0000 | 6.1913 | 0.8078 | 600 |
| nested | era2 | 1 | 5.9322 ± 0.0000 | 0.7827 ± 0.0000 | 7.0442 ± 0.0000 | 0.9680 ± 0.0000 | 5.8707 | 0.8129 | 600 |
| nested | era3 | 1 | 5.9277 ± 0.0000 | 0.8331 ± 0.0000 | 6.2326 ± 0.0000 | 0.9687 ± 0.0000 | 5.9821 | 0.8265 | 600 |
