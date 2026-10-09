"""Measure training and inference speed on this machine (measured, not extrapolated).

    python scripts/benchmark_speed.py --config configs/main.yaml

Reports training iterations per second at the configured batch size and the
median per-message encode / decode latency at every prefix length, and turns
the training speed into wall-clock estimates for the configured experiments.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from dnsc.channel import Channel
from dnsc.cli import make_parser, parse
from dnsc.codec import build_codec
from dnsc.train import sample_grans
from dnsc.utils import environment, get_device, save_json

SENTENCE = "The base station allocates resource blocks to the user equipment in every slot under fading."


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=30)
    cfg, args = parse(parser)
    device = get_device(cfg["device"])
    codec = build_codec(cfg, device)
    channel = Channel("awgn", 0)
    rng = np.random.default_rng(0)
    batch_size = int(cfg["pretrain"]["batch_size"])
    params = [p for p in codec.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    batch = codec.tokenize([SENTENCE] * batch_size)

    def train_step() -> None:
        z = codec.encode(batch)
        loss = sum(codec.loss_at(z, batch, m, 10.0, channel)[0].mean() for m in sample_grans(codec.grans, int(cfg["pretrain"]["grans_per_step"]), rng))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    for _ in range(3):  # warm-up
        train_step()
    sync(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        train_step()
    sync(device)
    it_per_s = args.steps / (time.perf_counter() - start)

    codec.eval()
    one = codec.tokenize([SENTENCE])
    latency = {}
    with torch.no_grad():
        enc_times = []
        for _ in range(args.repeats):
            sync(device)
            t0 = time.perf_counter()
            z = codec.encode(one)
            sync(device)
            enc_times.append(time.perf_counter() - t0)
        for m in codec.grans:
            dec_times = []
            y, _ = codec.transmit(z, m, 10.0, channel)
            for _ in range(max(5, args.repeats // 3)):
                sync(device)
                t0 = time.perf_counter()
                codec.generate(codec.prompts(y, m), max_new=int(one.ids.shape[1]) + 4)
                sync(device)
                dec_times.append(time.perf_counter() - t0)
            latency[str(m)] = {"decode_ms": 1e3 * float(np.median(dec_times)), "payload_bytes_fp32": 4 * m}
    encode_ms = 1e3 * float(np.median(enc_times))

    pre_h = int(cfg["pretrain"]["steps"]) / it_per_s / 3600
    scfg = cfg["stream"]
    stream_steps = len(scfg["domains"]) * int(scfg["cycles"]) * int(scfg["tau_slow"]) if scfg["kind"] == "hier" else int(scfg["temporal_steps"])
    stream_h = stream_steps / it_per_s / 3600
    print(f"\ndevice                : {device}")
    print(f"training speed        : {it_per_s:.2f} it/s at batch size {batch_size}")
    print(f"encode latency        : {encode_ms:.1f} ms per message (independent of m: the encoder runs in full)")
    for m, row in latency.items():
        print(f"decode latency  m={m:>4}: {row['decode_ms']:.1f} ms  (payload {row['payload_bytes_fp32']} B at fp32)")
    print(f"estimated pretraining : {pre_h:.2f} h for {cfg['pretrain']['steps']} steps")
    print(f"estimated stream run  : {stream_h:.2f} h per adaptive (method, seed) for {stream_steps} steps")
    out = Path(cfg["out_dir"]) / "eval" / "speed.json"
    save_json(out, {"device": str(device), "it_per_s": it_per_s, "encode_ms": encode_ms, "latency": latency, "pretrain_hours": pre_h, "stream_hours": stream_h, "env": environment()})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
