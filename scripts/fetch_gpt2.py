"""Build the pretrained GPT-2 (124M) backbone from files hosted on GitHub.

    python scripts/fetch_gpt2.py --out models/gpt2

For machines that cannot reach Hugging Face. The weights are the official
OpenAI GPT-2 small weights as published in the ONNX Model Zoo (the Hugging
Face ``gpt2`` checkpoint exported to ONNX; Git LFS object, SHA-256 checked);
the tokenizer is the original ``encoder.json`` / ``vocab.bpe`` pair. The script
copies the tensors, which keep their PyTorch names in the ONNX file, into a
``GPT2LMHeadModel`` and saves model and tokenizer in the usual format, so that
``backbone.name=models/gpt2`` works like any other backbone.

Where Hugging Face is reachable none of this is needed: use
``backbone.name=openai-community/gpt2``, which holds the same weights.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import urllib.request
from pathlib import Path

import numpy as np
import torch

ONNX_URL = "https://media.githubusercontent.com/media/onnx/models/main/validated/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx"
ONNX_SHA256 = "12fbb1ec2d2d70c8ebd21a290a348a4109447b98582af64c6f93b6526d5c8f35"
VOCAB_URL = "https://raw.githubusercontent.com/latitudegames/GPT-3-Encoder/master/encoder.json"
MERGES_URL = "https://raw.githubusercontent.com/latitudegames/GPT-3-Encoder/master/vocab.bpe"
CHECK_TEXT = "The European Parliament adopted the resolution on energy policy by a large majority."

from dnsc.cli import utf8_console


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"[gpt2] downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "dnsc-research/0.2"})
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(request, timeout=600) as response, open(tmp, "wb") as handle:
        while chunk := response.read(1 << 22):
            handle.write(chunk)
    tmp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="models/gpt2", help="directory for the converted model")
    parser.add_argument("--cache", default="", help="directory for the downloaded files (default: <out>/download)")
    args = parser.parse_args()
    out = Path(args.out)
    cache = Path(args.cache) if args.cache else out / "download"
    cache.mkdir(parents=True, exist_ok=True)

    onnx_path, vocab_path, merges_path = cache / "gpt2-lm-head-10.onnx", cache / "encoder.json", cache / "vocab.bpe"
    download(ONNX_URL, onnx_path)
    download(VOCAB_URL, vocab_path)
    download(MERGES_URL, merges_path)
    found = sha256(onnx_path)
    if found != ONNX_SHA256:
        raise SystemExit(f"checksum mismatch for {onnx_path}: {found}")

    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:
        raise SystemExit("this script needs the 'onnx' package: python -m pip install onnx") from exc
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    graph = onnx.load(str(onnx_path)).graph
    tensors = {t.name: t for t in graph.initializer}
    model = GPT2LMHeadModel(GPT2Config())  # the defaults are GPT-2 small
    state = model.state_dict()
    loaded = {}
    for name, target in state.items():
        if name not in tensors:
            continue
        value = torch.from_numpy(np.array(numpy_helper.to_array(tensors[name])))
        if value.shape != target.shape:
            raise SystemExit(f"shape mismatch for {name}: {tuple(value.shape)} vs {tuple(target.shape)}")
        loaded[name] = value.to(target.dtype)
    # the output layer is tied to the token embedding; in the ONNX graph it is a separate, transposed copy
    head = [t for t in graph.initializer if list(t.dims) == [768, 50257]]
    if len(head) != 1 or not np.array_equal(numpy_helper.to_array(head[0]).T, loaded["transformer.wte.weight"].numpy()):
        raise SystemExit("the output layer of the ONNX file is not the tied token embedding")
    loaded["lm_head.weight"] = loaded["transformer.wte.weight"]
    missing = [k for k in state if k not in loaded and not k.endswith((".attn.bias", ".attn.masked_bias"))]
    if missing:
        raise SystemExit(f"weights missing from the ONNX file: {missing[:5]}")
    model.load_state_dict(loaded, strict=False)
    model.eval()

    core = Tokenizer(models.BPE.from_file(str(vocab_path), str(merges_path)))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.decoder = decoders.ByteLevel()
    end = "<|endoftext|>"
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, bos_token=end, eos_token=end, unk_token=end)
    if len(tokenizer) != 50257 or tokenizer("Hello world")["input_ids"] != [15496, 995] or tokenizer.eos_token_id != 50256:
        raise SystemExit("the tokenizer files do not give the GPT-2 vocabulary")
    ids = tokenizer(CHECK_TEXT, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        full = torch.cat([torch.tensor([[tokenizer.eos_token_id]]), ids], dim=1)
        logits = model(full).logits[:, :-1]
        nll = torch.nn.functional.cross_entropy(logits[0], ids[0])
    perplexity = math.exp(float(nll))
    print(f"[gpt2] {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters; check sentence: {ids.shape[1]} tokens, perplexity {perplexity:.1f}")
    if not perplexity < 200:
        raise SystemExit("the converted model does not behave like a trained language model")

    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"[gpt2] saved to {out}  ->  use  backbone.name={out.as_posix()}")


if __name__ == "__main__":
    utf8_console()  # Windows consoles default to a legacy code page; the output contains ± and −
    main()
