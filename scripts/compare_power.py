"""Seed-0 stream runs with refinement rounds at unit power (decoder.refine_power=subset)
against the published runs of the same seed (gain of the whole message).

    python scripts/compare_power.py --new runs/v05_power_check/stream --old runs/six/stream --out results/refine_power

Reports, for the frozen receiver and DN-SC, the teacher-forced cost at 90 % token
accuracy (fixed prefix, refinement with signaling at its entropy) and the real-decoding
saving of DN-SC against the frozen fixed prefix on the audited steps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dnsc.linkbudget import audit_comparison

ROOT = Path(__file__).resolve().parent.parent


def load(folder: Path, method: str, seed: int) -> dict:
    return json.load(open(folder / f"hier_{method}_s{seed}.json", encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--new", default="runs/v05_power_check/stream")
    parser.add_argument("--old", default="runs/six/stream")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/refine_power")
    args = parser.parse_args()
    rows, out = [], {}
    for label, folder in (("gain of the whole message (published)", ROOT / args.old), ("unit power per round", ROOT / args.new)):
        fz, ns = load(folder, "frozen", args.seed), load(folder, "nested", args.seed)
        ref = fz["summary"]["symbols_at_target_fixed"]
        tf = {
            "frozen_fixed": ref,
            "frozen_refine": fz["summary"]["symbols_at_target_selective_fb_entropy"],
            "dnsc_fixed": ns["summary"]["symbols_at_target_fixed"],
            "dnsc_refine": ns["summary"]["symbols_at_target_selective_fb_entropy"],
        }
        real = audit_comparison({args.seed: ns}, {args.seed: fz}, 0.9, seed=20261015)
        real_f = audit_comparison({args.seed: fz}, {args.seed: fz}, 0.9, seed=20261015)
        out[label] = {"tf": tf, "real_dnsc_vs_frozen_fixed": real["vs_reference_entropy"], "real_frozen_refine_vs_frozen_fixed": real_f["vs_reference_entropy"], "power": ns["summary"]["decoder"].get("refine_power", "message")}
        rows.append(
            f"| {label} | {tf['frozen_fixed']:.3f} | {tf['frozen_refine']:.3f} ({100 * (tf['frozen_refine'] / ref - 1):+.1f}%) | {tf['dnsc_fixed']:.3f} | {tf['dnsc_refine']:.3f} ({100 * (tf['dnsc_refine'] / ref - 1):+.1f}%) | "
            f"{100 * real_f['vs_reference_entropy']['estimate']:+.1f}% | {100 * real['vs_reference_entropy']['estimate']:+.1f}% |"
        )
    target = ROOT / args.out
    target.mkdir(parents=True, exist_ok=True)
    (target / f"stream_seed{args.seed}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    head = [
        f"**Hierarchical stream, seed {args.seed}: refinement rounds at the gain of the whole message (published runs) and at unit power over the tokens they carry; real symbols per token for 90% token accuracy, signaling at its entropy (changes against the frozen fixed prefix of the same run)**",
        "",
        "| refinement rounds | frozen, fixed | frozen + refinement (TF) | DN-SC, fixed | DN-SC + refinement (TF) | frozen + refinement, real | DN-SC + refinement, real |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    (target / f"stream_seed{args.seed}.md").write_text("\n".join(head + rows) + "\n", encoding="utf-8")
    print("\n".join(head + rows))


if __name__ == "__main__":
    main()
