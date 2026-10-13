**Selective refinement on the 'hier' stream by the shell sent in the first round: symbols per token for 90% token accuracy (n/a = the target is not reached on that curve). The paper's policy is the first row of every method; the other rows are a post-hoc exploration, not a tuned result**

| method | seeds | first shell (symbols) | forward only | + feedback (bitmap) | + feedback (entropy) |
| --- | --- | --- | --- | --- | --- |
| context_only | 1 | 2 | 5.114 | 6.498 | 6.417 |
| context_only | 1 | 4 | 5.674 | 6.657 | 6.610 |
| context_only | 1 | 6 | 6.626 | 7.335 | 7.104 |
| context_only | 1 | 8 | 8.352 | 8.974 | 8.598 |
| context_only | 1 | 12 | 12.064 | 12.642 | 12.132 |
| context_only | 1 | 16 | 16.000 | 16.000 | 16.000 |
| frozen | 1 | 2 | 6.038 | 7.639 | 7.435 |
| frozen | 1 | 4 | 6.331 | 7.460 | 7.424 |
| frozen | 1 | 6 | 7.075 | 7.870 | 7.734 |
| frozen | 1 | 8 | 8.492 | 9.130 | 8.804 |
| frozen | 1 | 12 | 12.092 | 12.670 | 12.182 |
| frozen | 1 | 16 | 16.000 | 16.000 | 16.000 |
| nested | 1 | 2 | 4.829 | 6.144 | 6.087 |
| nested | 1 | 4 | 5.462 | 6.397 | 6.336 |
| nested | 1 | 6 | 6.557 | 7.253 | 7.002 |
| nested | 1 | 8 | 8.309 | 8.925 | 8.532 |
| nested | 1 | 12 | 12.056 | 12.634 | 12.117 |
| nested | 1 | 16 | 16.000 | 16.000 | 16.000 |
| single_rate | 1 | 2 | 4.911 | 6.245 | 6.183 |
| single_rate | 1 | 4 | 5.531 | 6.481 | 6.424 |
| single_rate | 1 | 6 | 6.583 | 7.284 | 7.040 |
| single_rate | 1 | 8 | 8.326 | 8.945 | 8.559 |
| single_rate | 1 | 12 | 12.061 | 12.639 | 12.126 |
| single_rate | 1 | 16 | 16.000 | 16.000 | 16.000 |
