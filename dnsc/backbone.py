"""Backbone language model and tokenizer loading.

``tiny-random`` builds a small, randomly initialised Llama-architecture model
and a byte-level BPE tokenizer trained on the toy corpora. It needs no
download and exists for tests and the offline smoke run only. With
``tiny.lm_steps > 0`` the tiny model is first trained as a plain language
model on *all* toy domains, so that - like a real pretrained LLM - the frozen
backbone already knows the language of every domain and only the codec has to
be learned and adapted.

Any other name is treated as a Hugging Face causal-LM id (for example
``HuggingFaceTB/SmolLM2-360M`` or ``Qwen/Qwen2.5-0.5B``); both use
``q_proj``/``v_proj`` linear layers, which is what the LoRA injection targets.
"""

from __future__ import annotations

from pathlib import Path

import torch

TINY = "tiny-random"


def _build_tiny_tokenizer(vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    from .data import toy

    texts = [r["text"] for name in toy.toy_names() for r in toy.make_toy(name)]
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


def load_tokenizer(bcfg, saved_dir: str | Path | None = None):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if bcfg["name"] == TINY:
        if saved_dir is not None and Path(saved_dir).exists():
            tokenizer = PreTrainedTokenizerFast.from_pretrained(str(saved_dir))
        else:
            tokenizer = _build_tiny_tokenizer(int(bcfg["tiny"]["vocab"]))
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


def warm_up_tiny_lm(lm, tokenizer, steps: int, seed: int = 0, batch_size: int = 32, lr: float = 3e-3) -> float:
    """Next-token training of the tiny backbone on every toy domain.

    Returns the final training loss (nats per token)."""
    from .data import toy

    texts = [r["text"] for name in toy.toy_names() for r in toy.make_toy(name)]
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    rows = [ids + [eos] for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(lm.parameters(), lr=lr)
    lm.train()
    loss_value = float("nan")
    for _ in range(steps):
        pick = torch.randint(len(rows), (batch_size,), generator=generator).tolist()
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
        loss_value = loss.item()
    lm.eval()
    return loss_value
