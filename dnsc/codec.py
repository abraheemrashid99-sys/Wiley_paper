"""The doubly nested semantic codec.

Transmitter  sentence -> frozen LLM (+ "tx" LoRA) with S learnable memory
             tokens appended -> linear projection to L real symbols -> shell-
             wise power normalisation. Any prefix of the L symbols is a valid
             transmission (nesting in rate).
Channel      AWGN or Rayleigh block fading on the transmitted prefix.
Receiver     received prefix (zero-padded) -> de-projection to P soft-prompt
             tokens -> the same frozen LLM (+ "rx" LoRA) reconstructs the
             sentence autoregressively.

The parameters are exposed as named *groups* so that the multi-timescale
optimiser can assign each group its own update (and synchronisation) period
(nesting in time).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import TINY, load_lm, load_tokenizer, warm_up_tiny_lm
from .channel import Channel, shell_normalize
from .lora import inject_lora

GROUPS = ("tx_fast", "tx_lora", "rx_adapter", "rx_lora", "rx_static")


@dataclass
class TokenBatch:
    ids: torch.Tensor  # (B, T) right-padded, EOS-terminated
    lens: torch.Tensor  # (B,)

    def to(self, device) -> "TokenBatch":
        return TokenBatch(self.ids.to(device), self.lens.to(device))


@dataclass
class Decoded:
    texts: list[str]
    token_ids: list[list[int]]  # emitted tokens, cut before EOS
    mean_logprob: torch.Tensor  # (B,) mean log-probability of the emitted tokens
    mean_entropy: torch.Tensor  # (B,) mean predictive entropy (nats)
    min_prob: torch.Tensor  # (B,) smallest emitted-token probability


class SemanticCodec(nn.Module):
    def __init__(self, lm, tokenizer, ccfg, train_backbone: bool = False):
        super().__init__()
        self.lm = lm
        self.tok = tokenizer
        self.grans = [int(m) for m in ccfg["granularities"]]
        self.L = self.grans[-1]
        self.S = int(ccfg["n_slots"])
        self.P = int(ccfg["n_prompts"])
        self.max_tokens = int(ccfg["max_tokens"])
        self.train_backbone = bool(train_backbone)
        hidden = int(lm.config.hidden_size)
        self.hidden = hidden

        for param in lm.parameters():
            param.requires_grad_(self.train_backbone)
        lora = ccfg["lora"]
        self.loras = inject_lora(
            lm,
            list(lora["targets"]),
            int(lora["r"]),
            float(lora["alpha"]),
            float(lora["dropout"]),
            int(lora["tx_last_n"]),
            int(lora["rx_last_n"]),
        )
        std = float(lm.get_input_embeddings().weight.detach().float().std())

        # transmitter head
        self.mem_tokens = nn.Parameter(torch.randn(self.S, hidden) * std)
        self.tx_norm = nn.LayerNorm(hidden)
        self.tx_proj = nn.Linear(self.S * hidden, self.L)

        # receiver head: a static de-projection plus a low-rank adapter
        rank = int(ccfg["deproj_rank"])
        self.rx_deproj = nn.Linear(self.L, self.P * hidden)
        self.rx_dA = nn.Parameter(torch.empty(rank, self.L))
        nn.init.kaiming_uniform_(self.rx_dA, a=math.sqrt(5))
        self.rx_dB = nn.Parameter(torch.zeros(self.P * hidden, rank))
        self.rx_gran = nn.Embedding(len(self.grans), self.P * hidden)
        nn.init.zeros_(self.rx_gran.weight)
        self.rx_norm = nn.LayerNorm(hidden)
        self.prompt_scale = nn.Parameter(torch.tensor(std))

    # ------------------------------------------------------------------ utils
    @property
    def device(self) -> torch.device:
        return self.mem_tokens.device

    def _side(self, side: str | None) -> None:
        for layer in self.loras:
            layer.active = side

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm.get_input_embeddings()(ids)

    def _head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm.get_output_embeddings()(hidden)

    def groups(self) -> dict[str, list[tuple[str, nn.Parameter]]]:
        """Named parameter groups used by the multi-timescale optimiser."""
        out: dict[str, list[tuple[str, nn.Parameter]]] = {g: [] for g in GROUPS}
        for name, param in self.named_parameters():
            if "lora_A.tx" in name or "lora_B.tx" in name:
                out["tx_lora"].append((name, param))
            elif "lora_A.rx" in name or "lora_B.rx" in name:
                out["rx_lora"].append((name, param))
            elif name.startswith("lm."):
                continue  # frozen backbone
            elif name in ("rx_dA", "rx_dB"):
                out["rx_adapter"].append((name, param))
            elif name.startswith(("mem_tokens", "tx_norm", "tx_proj")):
                out["tx_fast"].append((name, param))
            else:
                out["rx_static"].append((name, param))
        return out

    def freeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------- tokenising
    def tokenize(self, texts: list[str]) -> TokenBatch:
        eos, pad = self.tok.eos_token_id, self.tok.pad_token_id
        encoded = self.tok(list(texts), add_special_tokens=False)["input_ids"]
        rows = [list(ids)[: self.max_tokens - 1] + [eos] for ids in encoded]
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), pad, dtype=torch.long)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        lens = torch.tensor([len(r) for r in rows], dtype=torch.long)
        return TokenBatch(ids, lens).to(self.device)

    # ------------------------------------------------------------ transmitter
    def encode(self, batch: TokenBatch) -> torch.Tensor:
        """Sentence -> L shell-normalised channel symbols, shape (B, L)."""
        self._side("tx")
        ids, lens = batch.ids, batch.lens
        bsz, width = ids.shape
        emb = self._embed(ids)
        mem = self.mem_tokens.to(emb.dtype).unsqueeze(0).expand(bsz, -1, -1)
        # place the S memory tokens directly after each sentence's last token
        pos = lens.unsqueeze(1) + torch.arange(self.S, device=ids.device).unsqueeze(0)
        index = pos.unsqueeze(-1).expand(-1, -1, self.hidden)
        x = torch.cat([emb, emb.new_zeros(bsz, self.S, self.hidden)], dim=1).scatter(1, index, mem)
        mask = (torch.arange(width + self.S, device=ids.device).unsqueeze(0) < (lens + self.S).unsqueeze(1)).long()
        hidden = self.lm.base_model(inputs_embeds=x, attention_mask=mask).last_hidden_state
        slots = hidden.gather(1, index)
        z = self.tx_proj(self.tx_norm(slots).flatten(1))
        return shell_normalize(z, self.grans)

    # ---------------------------------------------------------------- channel
    def transmit(self, z: torch.Tensor, m: int, snr_db, channel: Channel) -> tuple[torch.Tensor, torch.Tensor]:
        """Send the first ``m`` symbols; return the zero-padded observation."""
        y, eff = channel(z, snr_db)
        keep = (torch.arange(self.L, device=z.device) < m).to(y.dtype).unsqueeze(0)
        return y * keep, eff

    def mask_prefix(self, y_full: torch.Tensor, m: int) -> torch.Tensor:
        keep = (torch.arange(self.L, device=y_full.device) < m).to(y_full.dtype).unsqueeze(0)
        return y_full * keep

    # --------------------------------------------------------------- receiver
    def prompts(self, y: torch.Tensor, m: int) -> torch.Tensor:
        gran = self.grans.index(int(m))
        flat = self.rx_deproj(y) + F.linear(F.linear(y, self.rx_dA), self.rx_dB) + self.rx_gran.weight[gran]
        return self.rx_norm(flat.view(y.shape[0], self.P, self.hidden)) * self.prompt_scale

    def decode_tf(self, prompts: torch.Tensor, batch: TokenBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced decoding. Returns per-example (cross-entropy in nats
        per token, token accuracy)."""
        self._side("rx")
        ids, lens = batch.ids, batch.lens
        bsz, width = ids.shape
        x = torch.cat([prompts, self._embed(ids[:, :-1]).to(prompts.dtype)], dim=1)
        positions = torch.arange(self.P + width - 1, device=ids.device).unsqueeze(0)
        mask = (positions < (self.P + lens - 1).unsqueeze(1)).long()
        hidden = self.lm.base_model(inputs_embeds=x, attention_mask=mask).last_hidden_state
        logits = self._head(hidden[:, self.P - 1 :])  # position P-1+j predicts token j
        valid = (torch.arange(width, device=ids.device).unsqueeze(0) < lens.unsqueeze(1)).to(logits.dtype)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids.reshape(-1), reduction="none").view(bsz, width)
        ce = (ce * valid).sum(1) / lens
        acc = ((logits.argmax(-1) == ids).to(valid.dtype) * valid).sum(1) / lens
        return ce, acc

    @torch.no_grad()
    def generate(self, prompts: torch.Tensor, max_new: int | None = None) -> Decoded:
        """Greedy decoding with a key-value cache, plus confidence statistics."""
        self._side("rx")
        max_new = max_new or self.max_tokens
        bsz = prompts.shape[0]
        eos = self.tok.eos_token_id
        mask = torch.ones(bsz, self.P, dtype=torch.long, device=prompts.device)
        out = self.lm.base_model(inputs_embeds=prompts, attention_mask=mask, use_cache=True)
        past, hidden = out.past_key_values, out.last_hidden_state[:, -1]
        done = torch.zeros(bsz, dtype=torch.bool, device=prompts.device)
        tokens: list[torch.Tensor] = []
        sum_lp = torch.zeros(bsz, device=prompts.device)
        sum_ent = torch.zeros(bsz, device=prompts.device)
        min_lp = torch.zeros(bsz, device=prompts.device)
        count = torch.zeros(bsz, device=prompts.device)
        for _ in range(max_new):
            logp = F.log_softmax(self._head(hidden).float(), dim=-1)
            best_lp, token = logp.max(dim=-1)
            entropy = -(logp.exp() * logp).sum(-1)
            live = (~done).float()
            sum_lp += best_lp * live
            sum_ent += entropy * live
            min_lp = torch.minimum(min_lp, best_lp * live)
            count += live
            tokens.append(torch.where(done, torch.full_like(token, eos), token))
            done = done | (token == eos)
            if bool(done.all()):
                break
            mask = torch.cat([mask, mask.new_ones(bsz, 1)], dim=1)
            out = self.lm.base_model(
                inputs_embeds=self._embed(token).unsqueeze(1).to(prompts.dtype),
                attention_mask=mask,
                past_key_values=past,
                use_cache=True,
            )
            past, hidden = out.past_key_values, out.last_hidden_state[:, -1]
        stacked = torch.stack(tokens, dim=1).cpu().tolist()
        texts, kept = [], []
        for row in stacked:
            cut = row.index(eos) if eos in row else len(row)
            kept.append(row[:cut])
            texts.append(self.tok.decode(row[:cut], skip_special_tokens=True).strip())
        count = count.clamp_min(1.0)
        return Decoded(texts, kept, (sum_lp / count).cpu(), (sum_ent / count).cpu(), min_lp.exp().cpu())

    # ------------------------------------------------------------ convenience
    def loss_at(self, z: torch.Tensor, batch: TokenBatch, m: int, snr_db, channel: Channel):
        y, _ = self.transmit(z, m, snr_db, channel)
        ce, acc = self.decode_tf(self.prompts(y, m), batch)
        return ce, acc

    @torch.no_grad()
    def reconstruct(self, texts: list[str], m: int, snr_db, channel: Channel) -> Decoded:
        batch = self.tokenize(texts)
        y, _ = self.transmit(self.encode(batch), m, snr_db, channel)
        return self.generate(self.prompts(y, m), max_new=int(batch.ids.shape[1]) + 4)

    # ---------------------------------------------------------- checkpointing
    def codec_state(self) -> dict[str, torch.Tensor]:
        """Everything that is not frozen pretrained backbone weight."""
        keep_backbone = self.train_backbone or getattr(self, "_always_save_backbone", False)
        state = {}
        for name, tensor in self.state_dict().items():
            if not name.startswith("lm.") or "lora_" in name or keep_backbone:
                state[name] = tensor.detach().cpu().clone()
        return state


def build_codec(cfg, device: torch.device, tokenizer_dir: str | Path | None = None, warm: bool = True) -> SemanticCodec:
    tokenizer = load_tokenizer(cfg["backbone"], tokenizer_dir)
    lm = load_lm(cfg["backbone"], tokenizer)
    lm_steps = int(cfg["backbone"]["tiny"].get("lm_steps", 0)) if cfg["backbone"]["name"] == TINY else 0
    if warm and lm_steps > 0:  # stand-in for a pretrained LLM; see dnsc/backbone.py
        final = warm_up_tiny_lm(lm, tokenizer, lm_steps, seed=int(cfg["seed"]))
        print(f"[backbone] tiny LM warmed up for {lm_steps} steps (loss {final:.3f} nats/token)")
    codec = SemanticCodec(lm, tokenizer, cfg["codec"], train_backbone=bool(cfg["backbone"]["train_backbone"]))
    # a randomly initialised backbone cannot be re-downloaded, so always store it
    codec._always_save_backbone = cfg["backbone"]["name"] == TINY
    return codec.to(device)


def save_codec(codec: SemanticCodec, directory: str | Path, extra: dict | None = None) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"state": codec.codec_state(), "extra": extra or {}}, directory / "codec.pt")
    codec.tok.save_pretrained(str(directory / "tokenizer"))


def load_codec(cfg, directory: str | Path, device: torch.device) -> SemanticCodec:
    directory = Path(directory)
    if not (directory / "codec.pt").exists():
        raise FileNotFoundError(f"no checkpoint at {directory}/codec.pt - run scripts/pretrain.py first")
    codec = build_codec(cfg, device, tokenizer_dir=directory / "tokenizer", warm=False)
    payload = torch.load(directory / "codec.pt", map_location="cpu")
    missing, unexpected = codec.load_state_dict(payload["state"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]}")
    not_backbone = [k for k in missing if not k.startswith("lm.") or "lora_" in k]
    if not_backbone:
        raise RuntimeError(f"checkpoint is missing codec weights: {not_backbone[:5]}")
    return codec
