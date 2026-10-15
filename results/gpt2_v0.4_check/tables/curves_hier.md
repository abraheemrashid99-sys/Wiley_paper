**Operating curves on the 'hier' stream: token accuracy at every fixed prefix length, and (symbols per token / token accuracy) under selective refinement at every confidence threshold**

| method | acc m=2 | acc m=4 | acc m=6 | acc m=8 | acc m=12 | acc m=16 | τ=0.5 | τ=0.7 | τ=0.9 | τ=0.95 | τ=0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| context_only | 0.5111 | 0.6799 | 0.8188 | 0.9062 | 0.9746 | 0.9914 | 6.62 sym / 0.9014 | 7.07 sym / 0.9462 | 7.72 sym / 0.9779 | 8.06 sym / 0.9845 | 8.80 sym / 0.9896 |
| frozen | 0.4043 | 0.5958 | 0.7630 | 0.8755 | 0.9654 | 0.9885 | 6.88 sym / 0.8805 | 7.45 sym / 0.9370 | 8.21 sym / 0.9750 | 8.60 sym / 0.9820 | 9.40 sym / 0.9871 |
| nested | 0.5434 | 0.7027 | 0.8338 | 0.9141 | 0.9775 | 0.9924 | 6.56 sym / 0.9080 | 6.97 sym / 0.9505 | 7.58 sym / 0.9790 | 7.90 sym / 0.9853 | 8.63 sym / 0.9904 |
