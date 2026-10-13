"""The doubly nested semantic codec (token-level, LLM-prior decoding).

Transmitter  Each token is mapped to a codeword of ``M`` real symbols taken
             from a *semantic codebook*: a small trainable map of the frozen
             LLM's token embeddings. Every shell of the codebook meets the
             average-power constraint on its own, so any prefix of ``m``
             symbols per token is a valid transmission (nesting in rate). No
             LLM runs at the transmitter to encode.
Channel      AWGN or Rayleigh block fading.
Receiver     Bayesian decoding. For every position the receiver combines

                 log p(token | y, past)  ∝  log p_prior(token | what was decoded before)
                                            + log p(y | token)

             where the likelihood is the exact Gaussian likelihood of the
             received symbols under each codeword and the prior is the LLM,
             sharpened by memories that live at separated timescales (nesting
             in time):

             fast    the context: the messages of the same conversation that
                     the receiver has already decoded are placed in front of
                     the message being decoded - no gradient, no
                     synchronisation, every message;
             medium  a low-dimensional logit bias and the fusion weights;
             slow    LoRA adapters on the LLM, kept in a few *slots* so that
                     knowledge of one domain is not overwritten by the next.

             (An optional cache of recently decoded tokens can be mixed into
             the prior as well; it is off by default.)

Because the receiver always knows how many symbol groups arrived, it knows the
sentence length; no end-of-sentence token is transmitted.

This replaces an earlier sentence-level soft-prompt codec (commit
``cd8932d``), which needed far more training than a CPU can give.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import TINY, load_lm, load_tokenizer, tiny_corpora, warm_up_tiny_lm
from .channel import Channel, nested_power_normalize, shell_slices
from .lora import inject_lora

GROUPS = ("codebook", "rx_bias", "rx_lora")
_TINY_LM_CACHE: dict[str, dict] = {}  # trained tiny backbones of this process (tests build the same one many times)
PRIORS = ("llm", "unigram", "none")


def _length_groups(lengths: list[int], overhead: int = 16) -> list[list[int]]:
    """Partition row indices into groups of similar length (ascending). A row
    joins the current group unless padding the group up to it would cost more
    positions than ``overhead``, the rough price of one more forward call."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    groups: list[list[int]] = []
    for i in order:
        if groups and len(groups[-1]) * (lengths[i] - lengths[groups[-1][-1]]) <= overhead:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def join_context(sentences: list[list[int]], limit: int) -> list[int]:
    """The most recent whole sentences (token ids) that fit in ``limit``
    tokens, in order. A context never starts in the middle of a sentence,
    unless the last sentence alone is longer than the limit."""
    if limit <= 0 or not sentences:
        return []
    kept: list[list[int]] = []
    total = 0
    for sentence in reversed(sentences):
        if total + len(sentence) > limit:
            break
        kept.insert(0, sentence)
        total += len(sentence)
    if not kept:
        return list(sentences[-1][-limit:])
    return [token for sentence in kept for token in sentence]


def slot_group(slot: int) -> str:
    """Name of the parameter group that holds LoRA slot ``slot``."""
    return "rx_lora" if slot == 0 else f"rx_lora#{slot}"


@dataclass
class TokenBatch:
    ids: torch.Tensor  # (B, T) right-padded token ids
    lens: torch.Tensor  # (B,)

    @property
    def mask(self) -> torch.Tensor:
        """(B, T) boolean mask of real tokens."""
        return torch.arange(self.ids.shape[1], device=self.ids.device).unsqueeze(0) < self.lens.unsqueeze(1)

    def to(self, device) -> "TokenBatch":
        return TokenBatch(self.ids.to(device), self.lens.to(device))


@dataclass
class Decoded:
    texts: list[str]
    token_ids: list[list[int]]
    mean_logprob: torch.Tensor  # (B,) mean log posterior of the chosen tokens
    mean_entropy: torch.Tensor  # (B,) mean posterior entropy (nats)
    min_prob: torch.Tensor  # (B,) smallest posterior probability of a chosen token
    token_logprob: torch.Tensor  # (B, T) log posterior of each chosen token (padding: 0)
    token_entropy: torch.Tensor  # (B, T) posterior entropy at each position (padding: 0)


class TokenCache:
    """Fast memory: exponentially decayed counts of recently decoded tokens.

    It needs no gradient and no synchronisation - the receiver builds it from
    its own decisions - and tracks the bursty vocabulary of the current topic.
    """

    def __init__(self, vocab: int, decay: float = 0.98, min_mass: float = 50.0, device=None):
        self.counts = torch.zeros(vocab, device=device)
        self.decay, self.min_mass = float(decay), float(min_mass)

    def update(self, tokens: torch.Tensor) -> None:
        """Decay once, then add the tokens of one message batch (1-D ids)."""
        self.counts.mul_(self.decay)
        if tokens.numel():
            self.counts.add_(torch.bincount(tokens.reshape(-1).to(self.counts.device), minlength=self.counts.numel()).to(self.counts.dtype))

    def mix(self, log_prior: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Interpolate a normalised log-prior (..., V) with the cache.

        ``weight`` in (0, 1) is the asymptotic share of the cache; the share
        actually used grows with the mass observed, so an empty cache leaves
        the prior untouched."""
        mass = float(self.counts.sum())
        if mass <= 0.0:
            return log_prior
        mu = (weight * (mass / (mass + self.min_mass))).clamp(1e-6, 1 - 1e-6).to(log_prior.dtype)
        vocab = self.counts.numel()
        log_cache = torch.log((self.counts + 1.0 / vocab) / (mass + 1.0)).to(log_prior.dtype)
        return torch.logaddexp(log_prior + torch.log1p(-mu), log_cache + torch.log(mu))


class SemanticCodec(nn.Module):
    def __init__(self, lm, tokenizer, ccfg, cache_weight: float = 0.1):
        super().__init__()
        self.lm = lm
        self.tok = tokenizer
        self.grans = [int(m) for m in ccfg["granularities"]]
        self.M = self.grans[-1]
        self.max_tokens = int(ccfg["max_tokens"])
        self.peak = float(ccfg["peak_ratio"])
        hidden = int(lm.config.hidden_size)
        self.hidden = hidden
        self.vocab = int(lm.get_input_embeddings().weight.shape[0])

        for param in lm.parameters():
            param.requires_grad_(False)
        lora = ccfg["lora"]
        self.n_slots = max(1, int(lora.get("slots", 1)))
        sides = tuple(f"s{k}" for k in range(self.n_slots))
        self.loras = inject_lora(lm, list(lora["targets"]), int(lora["r"]), float(lora["alpha"]), float(lora["dropout"]), int(lora["last_n"]), sides)
        self.slot: int | None = 0  # the active slow-memory slot; None switches every adapter off

        # semantic codebook: a linear projection of the frozen token embeddings
        # plus a small non-linear correction that starts at zero
        width = int(ccfg["codebook_hidden"])
        self.cb_lin = nn.Linear(hidden, self.M, bias=False)
        self.cb_in = nn.Linear(hidden, width)
        self.cb_out = nn.Linear(width, self.M, bias=False)
        nn.init.zeros_(self.cb_out.weight)
        scale = float(lm.get_input_embeddings().weight.detach().float().std())
        self.register_buffer("emb_scale", torch.tensor(scale if scale > 0 else 1.0))

        # medium memory: fusion weights and a logit bias living in embedding space
        self.prior_weight = nn.Parameter(torch.tensor(1.0))
        self.lik_weight = nn.Parameter(torch.tensor(1.0))
        self.bias_u = nn.Parameter(torch.zeros(hidden))
        share = min(max(float(cache_weight), 1e-4), 1 - 1e-4)
        self.cache_logit = nn.Parameter(torch.tensor(math.log(share / (1 - share))))  # share of the fast memory

        self.register_buffer("log_unigram", torch.full((self.vocab,), -math.log(self.vocab)))
        self._codebook: torch.Tensor | None = None
        self.train_codebook = False  # True while the codebook itself is being optimised

    # ------------------------------------------------------------------ utils
    @property
    def device(self) -> torch.device:
        return self.bias_u.device

    def set_slot(self, slot: int | None) -> None:
        """Select the active LoRA slot (``None``: the frozen backbone alone)."""
        if slot is not None and not 0 <= slot < self.n_slots:
            raise ValueError(f"slot {slot} does not exist (the codec has {self.n_slots})")
        self.slot = slot

    def _activate(self) -> None:
        side = None if self.slot is None else f"s{self.slot}"
        for layer in self.loras:
            layer.active = side

    def _embed_weight(self) -> torch.Tensor:
        return self.lm.get_input_embeddings().weight

    def _head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm.get_output_embeddings()(hidden)

    @property
    def start_token(self) -> int:
        return self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id

    def groups(self) -> dict[str, list[tuple[str, nn.Parameter]]]:
        """Named parameter groups used by the multi-timescale optimiser:
        ``codebook``, ``rx_bias`` and one LoRA group per slot (``rx_lora`` for
        slot 0, ``rx_lora#k`` for slot k)."""
        out: dict[str, list[tuple[str, nn.Parameter]]] = {g: [] for g in GROUPS}
        for k in range(1, self.n_slots):
            out[slot_group(k)] = []
        for name, param in self.named_parameters():
            if "lora_A." in name or "lora_B." in name:
                out[slot_group(int(name.rsplit(".s", 1)[1]))].append((name, param))
            elif name.startswith("lm."):
                continue  # frozen backbone
            elif name.startswith("cb_"):
                out["codebook"].append((name, param))
            else:
                out["rx_bias"].append((name, param))
        return out

    def freeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------- tokenising
    def tokenize(self, texts: list[str], continued: list[bool] | None = None) -> TokenBatch:
        """Token ids of the messages. A message that continues a document
        (``continued``) is tokenised as running text, i.e. with the leading
        space it has after the sentence before it; a message that opens a
        document is not."""
        pad = self.tok.pad_token_id
        if continued is not None:
            texts = [" " + text if flag else text for text, flag in zip(texts, continued)]
        encoded = self.tok(list(texts), add_special_tokens=False)["input_ids"]
        rows = [list(ids)[: self.max_tokens] or [self.start_token] for ids in encoded]
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), pad, dtype=torch.long)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        lens = torch.tensor([len(r) for r in rows], dtype=torch.long)
        return TokenBatch(ids, lens).to(self.device)

    def history_ids(self, history: list[str], limit: int) -> list[int]:
        """Token ids of earlier sentences of a document as running text (the
        first one opens the document), kept within ``limit`` tokens."""
        if not history:
            return []
        batch = self.tokenize(history, [i > 0 for i in range(len(history))])
        return join_context([batch.ids[i, : int(batch.lens[i])].tolist() for i in range(len(history))], limit)

    def _prefix(self, contexts: list[list[int]] | None, bsz: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """What the LLM reads before a message: the start token followed by
        the context of each row, left-padded to a common width. Returns the
        ids and the attention mask, both (B, 1 + longest context)."""
        longest = max((len(c) for c in contexts), default=0) if contexts else 0
        ids = torch.full((bsz, 1 + longest), self.tok.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros(bsz, 1 + longest, dtype=torch.long, device=device)
        for row in range(bsz):
            context = list(contexts[row]) if contexts else []
            cells = [self.start_token] + context
            ids[row, longest + 1 - len(cells) :] = torch.tensor(cells, dtype=torch.long, device=device)
            mask[row, longest + 1 - len(cells) :] = 1
        return ids, mask

    # ---------------------------------------------------------------- codebook
    def _compute_codebook(self) -> torch.Tensor:
        emb = self._embed_weight().detach().to(self.cb_lin.weight.dtype) / self.emb_scale
        raw = self.cb_lin(emb) + self.cb_out(F.gelu(self.cb_in(emb)))
        return nested_power_normalize(raw, self.log_unigram.exp(), self.grans, self.peak)

    def codebook(self) -> torch.Tensor:
        """(V, M) normalised codewords, cached between calls.

        While the codebook is being optimised (``train_codebook``) the cached
        tensor carries gradients; the caller must ``invalidate_codebook()``
        after every backward pass, because the graph is then spent."""
        want_grad = self.train_codebook and torch.is_grad_enabled()
        if self._codebook is None or (want_grad and not self._codebook.requires_grad):
            if want_grad:
                self._codebook = self._compute_codebook()
            else:
                with torch.no_grad():
                    self._codebook = self._compute_codebook()
        return self._codebook

    def invalidate_codebook(self) -> None:
        self._codebook = None

    # ------------------------------------------------------------ transmitter
    def encode(self, batch: TokenBatch) -> torch.Tensor:
        """Token ids -> (B, T, M) channel symbols (padding positions are zero)."""
        symbols = self.codebook()[batch.ids]
        return symbols * batch.mask.unsqueeze(-1).to(symbols.dtype)

    def block_gains(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """Per-message, per-shell power-control gains, shape (B, M).

        Every transmitted block - one shell of one message - is scaled to unit
        average power per symbol, so the transmit power is exact whatever the
        message contains."""
        gains = torch.ones(x.shape[0], self.M, device=x.device, dtype=x.dtype)
        for sl in shell_slices(self.grans):
            energy = x[:, :, sl].pow(2).sum(dim=(1, 2)).clamp_min(1e-12)
            count = lens.to(x.dtype) * (sl.stop - sl.start)
            gains[:, sl] = (count / energy).sqrt().unsqueeze(1)
        return gains

    def transmit(self, x: torch.Tensor, lens: torch.Tensor, snr_db, channel: Channel) -> tuple[torch.Tensor, torch.Tensor]:
        """Send all M symbols of every token once; prefixes are read off later.

        Returns the equalised observation ``y`` (B, T, M) and the per-message,
        per-symbol-position noise precision ``prec`` (B, M), i.e. the inverse
        noise variance after power control, fading and equalisation."""
        bsz, width, m = x.shape
        gains = self.block_gains(x, lens)
        eff = channel.effective_snr_db(bsz, snr_db).to(device=x.device, dtype=x.dtype)
        prec = gains.pow(2) * torch.pow(10.0, eff / 10.0).unsqueeze(1)
        noise = channel.noise((bsz, width, m), prec.detach().rsqrt().cpu().unsqueeze(1))
        return x + noise.to(device=x.device, dtype=x.dtype), prec

    # --------------------------------------------------------------- receiver
    def loglik_w(self, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Gaussian log-likelihood of every codeword, shape (..., V), given
        received symbols ``y`` (..., m) and their noise precisions ``weight``
        (..., m) - zero marks a symbol that was not received - up to a term
        that does not depend on the codeword (it cancels in the posterior):

            -1/2 sum_j w_j (y_j - c_vj)^2  =  sum_j w_j y_j c_vj - 1/2 sum_j w_j c_vj^2 + const.

        Both sums are evaluated in a single matrix product."""
        code = self.codebook().to(y.dtype)[:, : y.shape[-1]]
        return torch.cat([weight * y, -0.5 * weight], dim=-1) @ torch.cat([code, code * code], dim=-1).t()

    def loglik(self, y: torch.Tensor, m, prec: torch.Tensor) -> torch.Tensor:
        """Log-likelihood of every codeword given the received prefix, shape
        (B, T, V); see :meth:`loglik_w` for the constant that is left out.

        ``m`` is either one prefix length for the whole batch or a (B, T)
        tensor of per-token prefix lengths; ``prec`` is the (B, M) noise
        precision returned by :meth:`transmit`."""
        prec = prec.to(device=y.device, dtype=y.dtype)
        if not isinstance(m, int):
            keep = (torch.arange(self.M, device=y.device).view(1, 1, -1) < m.unsqueeze(-1)).to(y.dtype)
            return self.loglik_w(y, prec.unsqueeze(1) * keep)
        return self.loglik_w(y[..., :m], prec[:, :m].unsqueeze(1).expand(-1, y.shape[1], -1))

    # ------------------------------------------- the same, on real tokens only
    # A stream step spends most of its time on vocabulary-sized tensors. The
    # "flat" views keep one row per non-padding token, (N, .), instead of one
    # per position of the padded batch.
    def flat_channel(self, batch: TokenBatch, y: torch.Tensor, prec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Token ids (N,), received symbols (N, M) and noise precisions (N, M)
        of the non-padding tokens."""
        rows = batch.mask.nonzero()[:, 0]
        return batch.ids[batch.mask], y[batch.mask], prec.to(device=y.device, dtype=y.dtype)[rows]

    def flat_loss(self, ids: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, m: int, log_prior: torch.Tensor) -> torch.Tensor:
        """Mean posterior cross-entropy (nats/token) at prefix length ``m``
        from flat inputs; ``log_prior`` is (N, V)."""
        return F.cross_entropy(log_prior + self.lik_weight * self.loglik_w(y[:, :m], weight[:, :m]), ids)

    def _adjust_prior(self, logits: torch.Tensor, cache: TokenCache | None) -> torch.Tensor:
        """LLM logits -> normalised log-prior after the medium and fast memories."""
        bias = (self._embed_weight().detach().to(logits.dtype) @ self.bias_u) / self.emb_scale
        log_prior = F.log_softmax(self.prior_weight * F.log_softmax(logits.float(), dim=-1) + bias, dim=-1)
        return cache.mix(log_prior, torch.sigmoid(self.cache_logit)) if cache is not None else log_prior

    def lm_logits_tf(self, batch: TokenBatch, contexts: list[list[int]] | None = None, flat: bool = False) -> torch.Tensor:
        """Teacher-forced LLM logits (B, T, V): position t sees the true tokens
        before t. This is the only place a stream step runs the LLM.

        ``contexts`` optionally gives, per message, the token ids of the text
        that precedes it (an empty list for none). With ``flat`` the output
        layer is applied to the non-padding positions only and the result is
        (N, V).

        Every row is the sequence ``start, context, message`` padded on the
        right. Rows of similar length are run together, so that a batch with
        one long row does not make the LLM process padding for all the others.
        """
        self._activate()
        bsz, width = batch.ids.shape
        device = batch.ids.device
        lens = batch.lens.tolist()
        rows = []
        for b in range(bsz):
            context = list(contexts[b]) if contexts else []
            rows.append([self.start_token] + context + batch.ids[b, : lens[b] - 1].tolist())
        hidden = [None] * bsz  # per row: (len_b, H) states that predict its tokens
        for group in _length_groups([len(r) for r in rows]):
            longest = max(len(rows[b]) for b in group)
            inputs = torch.full((len(group), longest), self.tok.pad_token_id, dtype=torch.long, device=device)
            mask = torch.zeros(len(group), longest, dtype=torch.long, device=device)
            for i, b in enumerate(group):
                inputs[i, : len(rows[b])] = torch.tensor(rows[b], dtype=torch.long, device=device)
                mask[i, : len(rows[b])] = 1
            states = self.lm.base_model(input_ids=inputs, attention_mask=mask).last_hidden_state
            for i, b in enumerate(group):
                hidden[b] = states[i, len(rows[b]) - lens[b] : len(rows[b])]
        if flat:
            return self._head(torch.cat(hidden, dim=0))
        padded = torch.zeros(bsz, width, hidden[0].shape[-1], dtype=hidden[0].dtype, device=device)
        for b in range(bsz):
            padded[b, : lens[b]] = hidden[b]
        return self._head(padded)

    def prior_tf(self, batch: TokenBatch, cache: TokenCache | None = None, prior: str = "llm", contexts: list[list[int]] | None = None, flat: bool = False) -> torch.Tensor:
        """Teacher-forced log-prior over the vocabulary, shape (B, T, V), or
        (N, V) over the non-padding tokens with ``flat``."""
        lead = (int(batch.mask.sum()),) if flat else tuple(batch.ids.shape)
        if prior == "none":
            return torch.zeros(*lead, self.vocab, device=batch.ids.device)
        if prior == "unigram":
            return self.log_unigram.expand(*lead, -1)
        return self._adjust_prior(self.lm_logits_tf(batch, contexts, flat), cache)

    def posterior(self, log_prior: torch.Tensor, loglik: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(log_prior + self.lik_weight * loglik, dim=-1)

    def score_tf(self, log_post: torch.Tensor, batch: TokenBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-example (cross-entropy in nats/token, token accuracy) and the
        (B, T) argmax decisions."""
        valid = batch.mask.to(log_post.dtype)
        ce = -log_post.gather(-1, batch.ids.unsqueeze(-1)).squeeze(-1)
        decided = log_post.argmax(-1)
        lens = batch.lens.to(log_post.dtype)
        acc = ((decided == batch.ids).to(valid.dtype) * valid).sum(1) / lens
        return (ce * valid).sum(1) / lens, acc, decided

    def loss_at(self, batch: TokenBatch, y: torch.Tensor, prec: torch.Tensor, m, log_prior: torch.Tensor):
        """Teacher-forced posterior loss at prefix length ``m``."""
        return self.score_tf(self.posterior(log_prior, self.loglik(y, m, prec)), batch)

    @torch.no_grad()
    def decode(self, y: torch.Tensor, m, prec: torch.Tensor, lens: torch.Tensor, prior: str = "llm", cache: TokenCache | None = None, contexts: list[list[int]] | None = None) -> Decoded:
        """Free-running greedy MAP decoding of (B, T, M) received symbols at
        prefix length(s) ``m``."""
        return self.decode_loglik(self.loglik(y, m, prec), lens, prior, cache, contexts)

    @torch.no_grad()
    def decode_loglik(self, loglik: torch.Tensor, lens: torch.Tensor, prior: str = "llm", cache: TokenCache | None = None, contexts: list[list[int]] | None = None) -> Decoded:
        """Greedy MAP decoding from a precomputed (B, T, V) log-likelihood."""
        if prior not in PRIORS:
            raise ValueError(f"unknown prior '{prior}'")
        bsz, width, _ = loglik.shape
        device = loglik.device
        loglik = self.lik_weight * loglik
        valid = (torch.arange(width, device=device).unsqueeze(0) < lens.unsqueeze(1)).float()
        tokens = torch.zeros(bsz, width, dtype=torch.long, device=device)
        logp = torch.zeros(bsz, width, device=device)
        entropy = torch.zeros(bsz, width, device=device)

        def decide(position: int, log_prior: torch.Tensor) -> torch.Tensor:
            post = F.log_softmax(log_prior + loglik[:, position], dim=-1)
            best, token = post.max(dim=-1)
            tokens[:, position], logp[:, position] = token, best
            entropy[:, position] = -(post.exp() * post).sum(-1)
            return token

        if prior != "llm":
            flat = self.log_unigram.view(1, -1) if prior == "unigram" else torch.zeros(1, self.vocab, device=device)
            for position in range(width):
                decide(position, flat)
        else:
            self._activate()
            feed, mask = self._prefix(contexts, bsz, device)
            padded = feed.shape[1] > 1
            position = (mask.cumsum(dim=1) - 1).clamp_min(0)  # counts real tokens only, as in the teacher-forced pass
            past = None
            for index in range(width):
                extra = {"position_ids": position} if padded else {}
                out = self.lm.base_model(input_ids=feed, attention_mask=mask, past_key_values=past, use_cache=True, **extra)
                past = out.past_key_values
                feed = decide(index, self._adjust_prior(self._head(out.last_hidden_state[:, -1]), cache)).unsqueeze(1)
                position = position[:, -1:] + 1
                mask = torch.cat([mask, mask.new_ones(bsz, 1)], dim=1)

        counts = lens.float().clamp_min(1.0)
        rows = [tokens[i, : int(lens[i])].tolist() for i in range(bsz)]
        texts = [self.tok.decode(row, skip_special_tokens=True).strip() for row in rows]
        min_logp = logp.masked_fill(valid == 0, float("inf")).min(dim=1).values
        return Decoded(
            texts,
            rows,
            ((logp * valid).sum(1) / counts).cpu(),
            ((entropy * valid).sum(1) / counts).cpu(),
            min_logp.exp().cpu(),
            (logp * valid).cpu(),
            (entropy * valid).cpu(),
        )

    @torch.no_grad()
    def reconstruct(self, texts: list[str], m: int, snr_db, channel: Channel, prior: str = "llm", cache: TokenCache | None = None, contexts: list[list[int]] | None = None, continued: list[bool] | None = None) -> Decoded:
        batch = self.tokenize(texts, continued)
        y, prec = self.transmit(self.encode(batch), batch.lens, snr_db, channel)
        return self.decode(y, m, prec, batch.lens, prior, cache, contexts)

    def codebook_power(self, batch: TokenBatch, m: int) -> float:
        """Average power per symbol of a batch's codewords at prefix ``m``
        *before* block power control (1 means the codebook constraint is met
        exactly on this text; the transmitted power is always exactly 1)."""
        return float(self.codebook()[batch.ids[batch.mask], :m].detach().pow(2).mean())

    def reference_texts(self, texts: list[str], continued: list[bool] | None = None) -> list[str]:
        """What a perfect link returns: the text after tokenise/detokenise
        (and truncation to ``max_tokens``). Fidelity is scored against this."""
        batch = self.tokenize(texts, continued)
        return [self.tok.decode(batch.ids[i, : int(batch.lens[i])].tolist(), skip_special_tokens=True).strip() for i in range(len(texts))]

    # ---------------------------------------------------------- checkpointing
    def codec_state(self) -> dict[str, torch.Tensor]:
        """Everything that is not frozen pretrained backbone weight."""
        keep_backbone = getattr(self, "_always_save_backbone", False)
        state = {}
        for name, tensor in self.state_dict().items():
            if not name.startswith("lm.") or "lora_" in name or keep_backbone:
                state[name] = tensor.detach().cpu().clone()
        return state

    def load_state_dict(self, *args, **kwargs):  # codeword cache follows the weights
        self.invalidate_codebook()
        return super().load_state_dict(*args, **kwargs)


def build_codec(cfg, device: torch.device, tokenizer_dir: str | Path | None = None, warm: bool = True) -> SemanticCodec:
    tokenizer = load_tokenizer(cfg["backbone"], tokenizer_dir, cfg["data"])
    lm = load_lm(cfg["backbone"], tokenizer)
    lm_steps = int(cfg["backbone"]["tiny"].get("lm_steps", 0)) if cfg["backbone"]["name"] == TINY else 0
    if warm and lm_steps > 0:  # stand-in for a pretrained LLM; see dnsc/backbone.py
        tiny = cfg["backbone"]["tiny"]
        max_tokens = int(cfg["codec"]["max_tokens"]) + int(cfg["context"]["max_tokens"])  # a message and its context
        key = json.dumps([dict(tiny), int(cfg["seed"]), max_tokens, dict(cfg["data"]) if tiny.get("corpora") else None], sort_keys=True, default=str)
        if key in _TINY_LM_CACHE:  # the same model was already trained in this process
            lm.load_state_dict(_TINY_LM_CACHE[key])
        else:
            final = warm_up_tiny_lm(
                lm,
                tokenizer,
                tiny_corpora(cfg["backbone"], cfg["data"]),
                lm_steps,
                seed=int(cfg["seed"]),
                lr=float(tiny.get("lm_lr", 3.0e-3)),
                focus=str(tiny.get("lm_focus", "")),
                other_share=float(tiny.get("lm_other_share", 1.0)),
                max_tokens=max_tokens,
            )
            _TINY_LM_CACHE[key] = {name: value.clone() for name, value in lm.state_dict().items()}
            print(f"[backbone] tiny LM trained for {lm_steps} steps (loss {final:.3f} nats/token)")
    codec = SemanticCodec(lm, tokenizer, cfg["codec"], cache_weight=float(cfg["cache"]["weight"]))
    # a randomly initialised backbone cannot be re-downloaded, so always store it
    codec._always_save_backbone = cfg["backbone"]["name"] == TINY
    return codec.to(device)


def save_codec(codec: SemanticCodec, directory: str | Path, extra: dict | None = None) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    codec.tok.save_pretrained(str(directory / "tokenizer"))
    tmp = directory / "codec.pt.tmp"  # written last and atomically: its presence marks a finished checkpoint
    torch.save({"state": codec.codec_state(), "extra": extra or {}}, tmp)
    os.replace(tmp, directory / "codec.pt")


def load_codec(cfg, directory: str | Path, device: torch.device) -> SemanticCodec:
    directory = Path(directory)
    if not (directory / "codec.pt").exists():
        raise FileNotFoundError(f"no checkpoint at {directory}/codec.pt - run scripts/pretrain.py first")
    codec = build_codec(cfg, device, tokenizer_dir=directory / "tokenizer", warm=False)
    payload = torch.load(directory / "codec.pt", map_location="cpu")
    # Adapters are never trained before deployment (they start as the identity: B = 0). A run may
    # therefore ask for adapters the checkpoint does not hold - more slots, or another rank - and
    # those keep their fresh initialisation, which depends only on the seed of the run. Everything
    # else (the codebook, the fusion weights, the token statistics) must be in the checkpoint.
    is_lora = lambda key: "lora_A." in key or "lora_B." in key  # noqa: E731
    own = codec.state_dict()
    reshaped = {key for key, value in payload["state"].items() if is_lora(key) and key in own and own[key].shape != value.shape}
    saved_slots = {key.rsplit(".s", 1)[1] for key in payload["state"] if is_lora(key)}
    missing, unexpected = codec.load_state_dict({key: value for key, value in payload["state"].items() if key not in reshaped}, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]} - was it written by the earlier sentence-level codec? Re-run scripts/pretrain.py --force")
    fresh = lambda key: is_lora(key) and (key in reshaped or (bool(saved_slots) and key.rsplit(".s", 1)[1] not in saved_slots))  # noqa: E731
    not_backbone = [k for k in missing if (not k.startswith("lm.") or "lora_" in k) and not fresh(k)]
    if not_backbone:
        raise RuntimeError(f"checkpoint is missing codec weights: {not_backbone[:5]}")
    if reshaped:
        print(f"[codec] {len(reshaped)} adapter tensors have another shape than in the checkpoint (different rank): freshly initialised")
    return codec
