**Selective refinement on the 'hier' stream by the shell sent in the first round: symbols per token for 90% token accuracy (n/a = the target is not reached on that curve). The paper's policy is the first row of every method; the other rows are a post-hoc exploration, not a tuned result**

| method | seeds | first shell (symbols) | forward only | + feedback (bitmap) | + feedback (entropy) |
| --- | --- | --- | --- | --- | --- |
| context_only | 1 | 2 | 5.088 | 6.466 | 6.388 |
| context_only | 1 | 4 | 5.661 | 6.640 | 6.591 |
| context_only | 1 | 6 | 6.624 | 7.332 | 7.100 |
| context_only | 1 | 8 | 8.351 | 8.972 | 8.596 |
| context_only | 1 | 12 | 12.064 | 12.642 | 12.132 |
| context_only | 1 | 16 | 16.000 | 16.000 | 16.000 |
| frozen | 1 | 2 | 6.038 | 7.639 | 7.435 |
| frozen | 1 | 4 | 6.331 | 7.460 | 7.424 |
| frozen | 1 | 6 | 7.075 | 7.870 | 7.734 |
| frozen | 1 | 8 | 8.492 | 9.130 | 8.804 |
| frozen | 1 | 12 | 12.092 | 12.670 | 12.182 |
| frozen | 1 | 16 | 16.000 | 16.000 | 16.000 |
| nested | 1 | 2 | 4.807 | 6.116 | 6.061 |
| nested | 1 | 4 | 5.448 | 6.379 | 6.316 |
| nested | 1 | 6 | 6.555 | 7.250 | 6.998 |
| nested | 1 | 8 | 8.308 | 8.925 | 8.531 |
| nested | 1 | 12 | 12.056 | 12.634 | 12.117 |
| nested | 1 | 16 | 16.000 | 16.000 | 16.000 |
