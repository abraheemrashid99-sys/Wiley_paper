**Equivalence tests against 'nested' on the 'hier' stream: two one-sided tests at 5% on the per-step paired differences (90% block-bootstrap interval inside the margin). The margins are fixed in the configuration before the runs; 'no' means equivalence was not shown, not that a difference was shown**

| method | seeds | prior bits/token: this − nested [90% CI] | prior bits/token: equivalent within ±0.05 | selective sym/token: this − nested [90% CI] | selective sym/token: equivalent within ±0.05 |
| --- | --- | --- | --- | --- | --- |
| context_only | 3 | +0.3725 [+0.3302, +0.4123] | no | +0.3006 [+0.2429, +0.3573] | no |
| ewc | 1 | +0.1229 [+0.0811, +0.1708] | no | +0.1078 [+0.0705, +0.1502] | no |
| frozen | 3 | +1.5375 [+1.3514, +1.7233] | no | +1.1474 [+0.9833, +1.3162] | no |
| full_online | 3 | -0.0061 [-0.0276, +0.0151] | yes | +0.0060 [-0.0110, +0.0223] | yes |
| nested_no_context | 1 | +1.1445 [+0.8867, +1.4008] | no | +0.9196 [+0.6950, +1.1488] | no |
| nested_no_slots | 3 | +0.0963 [+0.0677, +0.1294] | no | +0.0708 [+0.0527, +0.0912] | no |
| nested_two_level | 1 | +0.0216 [+0.0143, +0.0296] | yes | +0.0060 [-0.0012, +0.0130] | yes |
| single_rate | 3 | +0.1067 [+0.0803, +0.1372] | no | +0.0771 [+0.0605, +0.0957] | no |
