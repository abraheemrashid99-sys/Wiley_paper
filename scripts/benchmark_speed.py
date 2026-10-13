"""Measure speed on this machine (measured, not extrapolated).

    python scripts/benchmark_speed.py --config configs/laptop.yaml

Reports iterations per second for codebook pretraining and for the three
kinds of stream step (forward-only, medium memory only, full adaptation with
its second LLM pass), and
the per-message decode latency at every prefix length, and turns them into
wall-clock estimates. To compare thread counts on a CPU, add e.g.
``threads=4`` or ``threads=8``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from dnsc.channel import Channel
from dnsc.cli import make_parser, parse
from dnsc.codec import build_codec
from dnsc.train import codebook_loss, sample_grans
from dnsc.utils import count_params, environment, get_device, save_json

SENTENCE = "The base station allocates resource blocks to the user equipment in every slot under fading."


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def peak_rss_mb() -> float | None:
    """Peak resident memory of this process in MB (None where the platform
    offers no cheap way to read it)."""
    try:
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0  # bytes on macOS, kB on Linux
    except Exception:
        try:
            import psutil

            return psutil.Process().memory_info().peak_wset / (1024.0 * 1024.0)  # Windows
        except Exception:
            return None


def timed(fn, steps: int, device: torch.device) -> float:
    fn()  # warm-up
    sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    sync(device)
    return steps / (time.perf_counter() - start)


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--steps", type=int, default=5, help="timed steps per measurement")
    parser.add_argument("--repeats", type=int, default=5, help="timed repeats for the latency figures")
    cfg, args = parse(parser)
    device = get_device(cfg["device"])
    codec = build_codec(cfg, device)
    channel = Channel("awgn", 0)
    rng = np.random.default_rng(0)
    scfg = cfg["stream"]
    n_words = max(4, (int(cfg["data"]["min_words"]) + int(cfg["data"]["max_words"])) // 2)
    sentence = " ".join((SENTENCE.split() * 4)[:n_words])  # a sentence of typical length
    groups = codec.groups()
    m_op = int(scfg["m_op"])

    # 1. codebook pretraining step (no LLM pass)
    codebook_params = [p for _, p in groups["codebook"]]
    optimizer = torch.optim.AdamW(codebook_params, lr=1e-6)
    n_tokens = int(cfg["pretrain"]["batch_tokens"])
    codec.train_codebook = True

    def pretrain_step() -> None:
        ids = torch.tensor(rng.integers(0, codec.vocab, size=n_tokens), device=device)
        loss, _ = codebook_loss(codec, ids, torch.full((n_tokens,), 10.0), channel, sample_grans(codec.grans, int(cfg["pretrain"]["grans_per_step"]), rng))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        codec.invalidate_codebook()

    pretrain_its = timed(pretrain_step, args.steps, device)
    codec.train_codebook = False
    codec.invalidate_codebook()

    # 2. stream steps
    batch = codec.tokenize([sentence] * int(scfg["batch_size"]))
    bias, lora = [p for _, p in groups["rx_bias"]], [p for _, p in groups["rx_lora"]]

    def stream_step(train_bias: bool, train_lora: bool):
        def step() -> None:
            for p in bias:
                p.requires_grad_(train_bias)
            for p in lora:
                p.requires_grad_(train_lora)
            y, prec = codec.transmit(codec.encode(batch), batch.lens, 10.0, channel)
            if train_lora:  # the receiver is scored with its own (stale) adapter first
                with torch.no_grad():
                    codec.loss_at(batch, y, prec, m_op, codec._adjust_prior(codec.lm_logits_tf(batch), None))
            with torch.set_grad_enabled(train_lora):
                logits = codec.lm_logits_tf(batch)
            with torch.set_grad_enabled(train_bias or train_lora):
                ce, _, _ = codec.loss_at(batch, y, prec, m_op, codec._adjust_prior(logits, None))
                if train_bias or train_lora:
                    ce.mean().backward()
                    for p in bias + lora:
                        p.grad = None

        return step

    forward_its = timed(stream_step(False, False), args.steps, device)
    medium_its = timed(stream_step(True, False), args.steps, device)
    full_its = timed(stream_step(True, True), args.steps, device)
    codec.freeze_all()

    # 3. decode latency, one message
    codec.eval()
    one = codec.tokenize([sentence])
    latency = {}
    with torch.no_grad():
        y, prec = codec.transmit(codec.encode(one), one.lens, 10.0, channel)
        for prior in ("llm", "none"):
            for m in codec.grans:
                times = []
                for _ in range(args.repeats):
                    sync(device)
                    t0 = time.perf_counter()
                    codec.decode(y, m, prec, one.lens, prior)
                    sync(device)
                    times.append(time.perf_counter() - t0)
                latency[f"{prior}_m{m}"] = 1e3 * float(np.median(times))

    stream_steps = len(scfg["domains"]) * int(scfg["cycles"]) * int(scfg["tau_slow"]) if scfg["kind"] == "hier" else int(scfg["temporal_steps"])
    pre_min = int(cfg["pretrain"]["steps"]) / pretrain_its / 60
    tokens = int(one.lens[0])
    print(f"\ndevice                  : {device}  (CPU threads: {torch.get_num_threads()})")
    print(f"codebook pretraining    : {pretrain_its:.2f} it/s  ->  about {pre_min:.0f} min for {cfg['pretrain']['steps']} steps")
    print(f"stream step, no training: {forward_its:.2f} it/s  (frozen, context_only)")
    print(f"stream step, bias only  : {medium_its:.2f} it/s")
    print(f"stream step, bias + LoRA: {full_its:.2f} it/s  (nested, single_rate, ewc, full_online)")
    for m in codec.grans:
        print(f"decode latency  m={m:>3}   : LLM prior {latency[f'llm_m{m}']:.0f} ms, no prior {latency[f'none_m{m}']:.1f} ms  ({tokens} tokens, {4 * m * tokens} B payload at fp32)")
    print(f"estimated stream run    : {stream_steps / full_its / 60:.0f} min per adaptive (method, seed) for {stream_steps} steps, {stream_steps / forward_its / 60:.0f} min without training; probes come on top")
    out = Path(cfg["out_dir"]) / "eval" / "speed.json"
    save_json(
        out,
        {
            "device": str(device),
            "threads": torch.get_num_threads(),
            "pretrain_it_per_s": pretrain_its,
            "stream_forward_it_per_s": forward_its,
            "stream_medium_it_per_s": medium_its,
            "stream_full_it_per_s": full_its,
            "decode_ms": latency,
            "tokens": tokens,
            "parameters": {
                "frozen backbone": count_params(p for name, p in codec.named_parameters() if name.startswith("lm.") and "lora_" not in name),
                "codebook (trained once)": count_params(p for _, p in groups["codebook"]),
                "medium memory (fusion weights and logit bias)": count_params(p for _, p in groups["rx_bias"]),
                "slow memory (one LoRA slot)": count_params(p for _, p in groups["rx_lora"]),
            },
            "peak_rss_mb": peak_rss_mb(),
            "env": environment(),
        },
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
