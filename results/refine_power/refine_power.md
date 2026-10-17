**Refinement rounds with the gain of the whole message (as simulated on the streams) and with the gain of the requested tokens (power exactly one in every round): held-out AG News and TeleQnA sentences, frozen GPT-2 prior, first round of 6 symbols, teacher-forced request policy, AWGN with the SNR law of the stream; real symbols per token for 90% token accuracy, signaling at its entropy at the capacity of 10 dB**

| prior | messages | tokens | refinement-block power with the message gain at threshold 0.9 (symbol-weighted mean; 5th-95th percentile of blocks) | fixed prefix | refinement, message gain | refinement, subset gain | change from correcting the power |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_context | 600 | 10743 | 1.150 (0.52-2.05) | 8.469 | 7.484 (-11.6%) | 7.562 (-10.7%) | +1.0% |
| true_context | 600 | 10707 | 1.121 (0.49-2.02) | 7.919 | 7.086 (-10.5%) | 7.140 (-9.8%) | +0.8% |
