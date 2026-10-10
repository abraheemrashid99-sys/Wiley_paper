**Operating curves on the 'hier' stream: token accuracy at every fixed prefix length, and (symbols per token / token accuracy) under selective refinement at every confidence threshold**

| method | acc m=2 | acc m=4 | acc m=6 | acc m=8 | acc m=12 | acc m=16 | τ=0.5 | τ=0.7 | τ=0.9 | τ=0.95 | τ=0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| context_only | 0.5017 | 0.6656 | 0.8054 | 0.8954 | 0.9694 | 0.9892 | 4.30 sym / 0.7981 | 5.27 sym / 0.8970 | 6.54 sym / 0.9634 | 7.14 sym / 0.9761 | 8.33 sym / 0.9857 |
| frozen | 0.3988 | 0.5803 | 0.7470 | 0.8614 | 0.9595 | 0.9856 | 4.98 sym / 0.7728 | 6.06 sym / 0.8863 | 7.39 sym / 0.9591 | 8.00 sym / 0.9732 | 9.14 sym / 0.9830 |
| full_online | 0.5355 | 0.6918 | 0.8236 | 0.9065 | 0.9734 | 0.9906 | 4.09 sym / 0.8095 | 5.00 sym / 0.9021 | 6.23 sym / 0.9647 | 6.82 sym / 0.9772 | 8.02 sym / 0.9867 |
| nested | 0.5338 | 0.6912 | 0.8230 | 0.9059 | 0.9732 | 0.9905 | 4.09 sym / 0.8084 | 5.00 sym / 0.9011 | 6.23 sym / 0.9644 | 6.82 sym / 0.9768 | 8.01 sym / 0.9865 |
| nested_no_slots | 0.5350 | 0.6957 | 0.8288 | 0.9102 | 0.9766 | 0.9920 | 4.05 sym / 0.8103 | 4.93 sym / 0.9033 | 6.10 sym / 0.9658 | 6.67 sym / 0.9783 | 7.81 sym / 0.9883 |
| single_rate | 0.5285 | 0.6843 | 0.8176 | 0.9026 | 0.9722 | 0.9901 | 4.14 sym / 0.8063 | 5.07 sym / 0.9003 | 6.31 sym / 0.9640 | 6.90 sym / 0.9765 | 8.09 sym / 0.9862 |
