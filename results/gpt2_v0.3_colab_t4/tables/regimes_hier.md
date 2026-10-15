**The 'hier' stream by slow regime: prequential metrics while the regime is active, and held-out probes of it at the end of the stream (mean ± SD over seeds). A probe scores the receiver's copy at that moment: methods are comparable on a probe only at equal staleness (steps since the slowest level was shipped)**

| method | regime | seeds | prior bits/token | tok-acc @m_op | selective sym/token | selective tok-acc | held-out prior bits (end) | held-out tok-acc (end) | receiver staleness at the probe (steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| context_only | ag_news | 3 | 6.2151 ± 0.0435 | 0.8070 ± 0.0214 | 6.5066 ± 0.2836 | 0.9645 ± 0.0021 | 6.3421 | 0.8337 | - |
| context_only | teleqna | 3 | 5.0418 ± 0.0752 | 0.8114 ± 0.0246 | 6.4589 ± 0.4269 | 0.9600 ± 0.0055 | 4.8965 | 0.8614 | - |
| ewc | ag_news | 1 | 6.0855 ± 0.0000 | 0.8264 ± 0.0000 | 6.2889 ± 0.0000 | 0.9670 ± 0.0000 | 6.3994 | 0.8135 | 12 |
| ewc | teleqna | 1 | 4.7415 ± 0.0000 | 0.8398 ± 0.0000 | 5.9124 ± 0.0000 | 0.9657 ± 0.0000 | 4.3140 | 0.8895 | 12 |
| frozen | ag_news | 3 | 6.5965 ± 0.0351 | 0.7887 ± 0.0251 | 6.7880 ± 0.2874 | 0.9650 ± 0.0017 | 6.6539 | 0.8150 | - |
| frozen | teleqna | 3 | 6.9904 ± 0.1068 | 0.7166 ± 0.0374 | 7.8710 ± 0.5162 | 0.9523 ± 0.0065 | 7.2270 | 0.7648 | - |
| full_online | ag_news | 3 | 5.9677 ± 0.0347 | 0.8190 ± 0.0206 | 6.4030 ± 0.2782 | 0.9678 ± 0.0010 | 6.1575 | 0.8295 | 0 |
| full_online | teleqna | 3 | 4.5320 ± 0.0668 | 0.8362 ± 0.0200 | 5.9733 ± 0.3419 | 0.9619 ± 0.0049 | 4.1739 | 0.8976 | 0 |
| nested | ag_news | 3 | 5.9684 ± 0.0369 | 0.8193 ± 0.0206 | 6.4052 ± 0.2772 | 0.9681 ± 0.0011 | 5.8342 | 0.8503 | 150 |
| nested | teleqna | 3 | 4.5435 ± 0.0758 | 0.8351 ± 0.0208 | 5.9591 ± 0.3456 | 0.9606 ± 0.0053 | 4.1158 | 0.9004 | 150 |
| nested_no_context | ag_news | 1 | 6.3864 ± 0.0000 | 0.8160 ± 0.0000 | 6.4992 ± 0.0000 | 0.9691 ± 0.0000 | 6.2337 | 0.8201 | 150 |
| nested_no_context | teleqna | 1 | 6.4838 ± 0.0000 | 0.7649 ± 0.0000 | 7.3257 ± 0.0000 | 0.9630 ± 0.0000 | 6.4744 | 0.8180 | 150 |
| nested_no_slots | ag_news | 3 | 6.0580 ± 0.0255 | 0.8139 ± 0.0204 | 6.4460 ± 0.2728 | 0.9669 ± 0.0012 | 6.1574 | 0.8295 | 0 |
| nested_no_slots | teleqna | 3 | 4.6464 ± 0.0636 | 0.8310 ± 0.0211 | 6.0598 ± 0.3614 | 0.9614 ± 0.0049 | 4.1739 | 0.8976 | 0 |
| nested_two_level | ag_news | 1 | 5.9873 ± 0.0000 | 0.8325 ± 0.0000 | 6.2311 ± 0.0000 | 0.9676 ± 0.0000 | 5.8222 | 0.8459 | 110 |
| nested_two_level | teleqna | 1 | 4.6370 ± 0.0000 | 0.8446 ± 0.0000 | 5.7666 ± 0.0000 | 0.9634 ± 0.0000 | 4.1413 | 0.9012 | 110 |
| single_rate | ag_news | 3 | 6.0664 ± 0.0260 | 0.8134 ± 0.0202 | 6.4513 ± 0.2711 | 0.9667 ± 0.0014 | 6.1218 | 0.8331 | 12 |
| single_rate | teleqna | 3 | 4.6589 ± 0.0668 | 0.8302 ± 0.0203 | 6.0671 ± 0.3479 | 0.9612 ± 0.0054 | 4.1731 | 0.8960 | 12 |
