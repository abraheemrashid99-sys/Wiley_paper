# Power of the refinement rounds

The stream runs of versions 0.2–0.4 transmit every message once, with block
power control over the whole message, and read the refinement rounds off
that transmission. A refinement round carries its shell only for the
requested tokens; with the gain of the whole message its average power is
not one. Version 0.5 adds `decoder.refine_power=subset`, which scales every
round to unit power over the tokens it carries.

| File | Content |
| --- | --- |
| `refine_power.md` / `.json` | held-out AG News and TeleQnA sentences (600 messages, frozen prior with and without the true context): the power of the refinement rounds with the message gain, and the cost of refinement at 90 % token accuracy with either gain (`scripts/check_refine_power.py --messages 600`) |
| `stream_seed0.md` / `.json` | the hierarchical stream, seed 0, frozen receiver and DN-SC, repeated with `decoder.refine_power=subset` and compared with the published runs of the same seed (`scripts/compare_power.py`) |
| `stream/` | the two repeated runs (code revision recorded in each file) |

Result: on held-out text the rounds carried 1.15 and 1.12 times the allowed
power at threshold 0.9, and unit power raises the cost of refinement by
1.0 % and 0.8 %. On the stream (seed 0), the cost of DN-SC with refinement
rises by 0.4 % teacher-forced (saving against the frozen fixed prefix
23.0 % → 22.7 %; decoded for real 27.9 % either way) and that of the frozen
receiver with refinement by 1.1 % (14.9 % → 13.9 %; decoded for real
17.8 % → 16.6 %). The fixed prefixes are unaffected.

Command of the stream runs:

```
python scripts/run_stream.py --config configs/gpt2.yaml --methods frozen,nested --seeds 0 \
    --ckpt results/gpt2/base out_dir=runs/v05_power_check threads=1 stream.kind=hier decoder.refine_power=subset
```
