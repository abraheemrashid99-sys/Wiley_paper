"""Backbone language model and tokenizer loading.

``tiny-random`` builds a small, randomly initialised Llama-architecture model
and a byte-level BPE tokenizer, with no download. With ``tiny.lm_steps > 0``
the model is first trained as a plain language model and then frozen, so that
it plays the role of a pretrained LLM. By default it is trained on the toy
corpora (tests and the offline smoke run). ``tiny.corpora`` names real corpora
instead: this gives a rehearsal on natural text for machines that cannot
download a pretrained model. ``tiny.lm_focus`` / ``tiny.lm_other_share`` make
the model a good prior for one corpus and a poorer one for the others, as a
general LLM is for a specialised domain. Nothing produced with ``tiny-random``
is a result of the paper.

Any other name is treated as a Hugging Face causal-LM id (for example
``HuggingFaceTB/SmolLM2-135M`` or ``Qwen/Qwen2.5-0.5B``); both use
``q_proj``/``v_proj`` linear layers, which is what the LoRA injection targets.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

TINY = "tiny-random"


def tiny_corpora(bcfg, data_cfg=None) -> dict[str, list[list[str]]]:
    """Training text of the tiny backbone: documents (lists of sentences) by
    corpus name."""
    names = list(bcfg["tiny"].get("corpora") or [])
    if not names:
        from .data import toy

        out = {}
        for name in toy.toy_names():
            docs: dict[int, list[str]] = {}
            for record in toy.make_toy(name):
                docs.setdefault(record["doc"], []).append(record["text"])
            out[name] = list(docs.values())
        return out
    if data_cfg is None:
        raise ValueError("backbone.tiny.corpora needs the data config")
    from .data import get_corpus

    return {name: [[r["text"] for r in doc] for doc in get_corpus(name, data_cfg).documents("train")] for name in names}


def _build_tiny_tokenizer(vocab_size: int, texts: list[str]):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[EOS]", "[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")


def load_tokenizer(bcfg, saved_dir: str | Path | None = None, data_cfg=None):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if bcfg["name"] == TINY:
        if saved_dir is not None and Path(saved_dir).exists():
            tokenizer = PreTrainedTokenizerFast.from_pretrained(str(saved_dir))
        else:
            texts = [" ".join(doc) for group in tiny_corpora(bcfg, data_cfg).values() for doc in group]
            tokenizer = _build_tiny_tokenizer(int(bcfg["tiny"]["vocab"]), texts)
    else:
        tokenizer = AutoTokenizer.from_pretrained(bcfg["name"])
    if tokenizer.eos_token_id is None:
        raise RuntimeError(f"tokenizer of '{bcfg['name']}' defines no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_lm(bcfg, tokenizer):
    """Return the causal LM in float32 (the only dtype fully supported on MPS)."""
    if bcfg["name"] == TINY:
        from transformers import LlamaConfig, LlamaForCausalLM

        tiny = bcfg["tiny"]
        config = LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=int(tiny["hidden"]),
            intermediate_size=2 * int(tiny["hidden"]),
            num_hidden_layers=int(tiny["layers"]),
            num_attention_heads=int(tiny["heads"]),
            num_key_value_heads=int(tiny["heads"]),
            max_position_embeddings=256,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.eos_token_id,
        )
        lm = LlamaForCausalLM(config)
    else:
        from transformers import AutoModelForCausalLM

        lm = AutoModelForCausalLM.from_pretrained(bcfg["name"])
    return lm.to(torch.float32)


def warm_up_tiny_lm(lm, tokenizer, corpora: dict[str, list[list[str]]], steps: int, seed: int = 0, batch_size: int = 32, lr: float = 3e-3, focus: str = "", other_share: float = 1.0, max_tokens: int = 64) -> float:
    """Next-token training of the tiny backbone on the documents of ``corpora``.

    A training example is a document as running text (truncated to
    ``max_tokens`` tokens), so the model learns to use the sentences that
    came before, as a pretrained LLM does. With ``focus`` set, documents of
    the other corpora are sampled only with probability ``other_share``: the
    model then knows their vocabulary but is a poorer prior for them. Returns
    the mean training loss of the last steps (nats per token)."""
    texts, weights = [], []
    for name, group in corpora.items():
        share = 1.0 if (not focus or name == focus) else other_share
        chunks = [" ".join(doc[i : i + 6]) for doc in group for i in range(0, len(doc), 6)]  # long documents in pieces
        texts += chunks
        weights += [share / max(len(chunks), 1)] * len(chunks)
    probs = torch.tensor(weights, dtype=torch.float64)
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    rows = [ids[: max_tokens - 1] + [eos] for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(lm.parameters(), lr=lr)
    lm.train()
    recent: list[float] = []
    for step in range(steps):
        for group in optimizer.param_groups:  # linear warm-up, then cosine decay
            group["lr"] = lr * min(1.0, (step + 1) / max(1, steps // 20)) * (0.5 * (1.0 + math.cos(math.pi * step / steps)) if steps > 400 else 1.0)
        pick = torch.multinomial(probs, batch_size, replacement=True, generator=generator).tolist()
        width = max(len(rows[i]) for i in pick)
        ids = torch.full((batch_size, width), pad, dtype=torch.long)
        mask = torch.zeros(batch_size, width, dtype=torch.long)
        for row, i in enumerate(pick):
            ids[row, : len(rows[i])] = torch.tensor(rows[i])
            mask[row, : len(rows[i])] = 1
        context = torch.cat([torch.full((batch_size, 1), eos, dtype=torch.long), ids[:, :-1]], dim=1)
        logits = lm(input_ids=context, attention_mask=mask).logits
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids.reshape(-1), reduction="none")
        loss = (loss * mask.reshape(-1)).sum() / mask.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        recent = (recent + [loss.item()])[-20:]
        if steps >= 1000 and (step + 1) % 500 == 0:
            print(f"[backbone] tiny LM step {step + 1}/{steps} loss={sum(recent) / len(recent):.3f} nats/token", flush=True)
    lm.eval()
    return sum(recent) / max(len(recent), 1)
