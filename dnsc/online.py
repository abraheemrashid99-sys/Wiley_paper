"""Online adaptation on a multi-timescale drift stream (prequential protocol).

Every incoming batch is first *transmitted and scored* with the receiver as it
stands, and only then used for adaptation ("test-then-train"), so the reported
stream metrics are never contaminated by training on the scored data.

The receiver's language prior has three memories at separated timescales:

fast    context        the conversation so far       every message   free
medium  logit bias     fusion weights, vocabulary    every C_med     kilobits
slow    LoRA slots     one adapter per domain        every C_slow    megabits

The context is what the receiver itself decoded of the earlier messages of
the same conversation; it costs nothing. The gradient levels are trained on
the true text, which only the transmitter has. The transmitter therefore
keeps a working copy of them that learns from every message, and ships the
quantised change to the receiver once per period of the level; every shipment
is charged to the synchronisation ledger. What is scored is always the
*receiver's* copy, stale between two shipments. With several LoRA slots, the
transmitter also measures at regular intervals which slot (or none) predicts
the current text best and signals the choice; only the active slot learns.

Information pattern: the transmitter knows the text, the codebook, the
fed-back average SNR and its own working copies. It does not know the
receiver's noise or decisions, so it trains with the *true* earlier messages
as context and on a channel realisation of its own.

Selective refinement is scored on every message teacher-forced (each token
is judged given the true tokens before it), which costs one LLM pass per
step. Two things keep that emulation honest. The return-link signalling it
implies is counted for every threshold and every choice of the first shell
(``refine_grid``; see :class:`dnsc.refine.FeedbackMeter`). And every
``stream.live_every`` steps the same received symbols are decoded for real -
free-running, round by round - so the emulated and the real operating curve
are reported side by side (``live_curve``).

Methods
-------
frozen             no memory at all: every message is decoded on its own
context_only       the fast memory alone: zero synchronisation cost
single_rate        context, and both gradient levels shipped at one period
                   chosen so that the synchronisation budget equals the
                   nested method's; one LoRA adapter
ewc                single_rate plus an Elastic-Weight-Consolidation penalty
full_online        context, and both gradient levels shipped after every
                   message (adaptivity upper reference at a prohibitive cost)
nested             the proposal: context, bias at C_med, LoRA slots at C_slow
nested_no_slots    nested with a single LoRA adapter
nested_no_context  nested without the fast memory
nested_two_level   nested with bias and LoRA collapsed to one period
                   (matched budget)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .channel import Channel, capacity_real
from .codec import SemanticCodec, TokenCache, join_context, load_codec, slot_group
from .data import build_stream, get_corpus
from .data.stream import build_probes
from .linkbudget import net_semantic_rate, symbols_at_accuracy
from .metrics import expected_calibration_error, load_similarity
from .nested import Level, NestedOptimizer
from .refine import FeedbackMeter, feedback_bits_per_token, level_stats, progressive_decode, rounds_per_message, stop_levels
from .train import LN2, evaluate_gen, evaluate_tf, sample_grans
from .utils import Timer, environment, get_device, save_json, set_seed

METHODS = ("frozen", "context_only", "single_rate", "ewc", "full_online", "nested", "nested_no_slots", "nested_no_context", "nested_two_level")
NO_CONTEXT = ("frozen", "nested_no_context")  # methods that decode every message on its own
SLOTTED = ("nested", "nested_no_context", "nested_two_level")  # methods that keep several LoRA slots


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _dense_bits(params, bits: int, topk: float) -> int:
    total = 0
    for _, p in params:
        n = p.numel()
        if bits <= 0:
            total += 32 * n
        elif topk < 1.0:
            k = max(1, int(round(topk * n)))
            total += 32 + k * (bits + max(1, math.ceil(math.log2(n))))
        else:
            total += 32 + n * bits
    return total


def matched_period(codec: SemanticCodec, ncfg) -> int:
    """Period at which synchronising *all* gradient levels together costs the
    same bits per step as the nested schedule."""
    if ncfg["single_period"] != "auto":
        return int(ncfg["single_period"])
    groups = codec.groups()
    bits, topk = int(ncfg["sync"]["bits"]), float(ncfg["sync"]["topk"])
    b_medium = _dense_bits(groups["rx_bias"], bits, topk)
    b_slow = _dense_bits(groups["rx_lora"], bits, topk)
    if ncfg["adapt_codebook"]:
        b_slow += _dense_bits(groups["codebook"], bits, topk)
    nested_rate = b_medium / int(ncfg["periods"]["medium"]) + b_slow / int(ncfg["periods"]["slow"])
    return max(1, int(round((b_medium + b_slow) / nested_rate)))


def n_slots_of(codec: SemanticCodec, method: str) -> int:
    return codec.n_slots if method in SLOTTED else 1


def build_levels(codec: SemanticCodec, method: str, ncfg) -> list[Level]:
    """Gradient levels of a method (the context is handled separately)."""
    if method not in METHODS:
        raise ValueError(f"unknown method '{method}'; choose from {METHODS}")
    if method in ("frozen", "context_only"):
        return []
    groups = codec.groups()
    medium, slow = int(ncfg["periods"]["medium"]), int(ncfg["periods"]["slow"])
    lr_medium, lr_slow = float(ncfg["lrs"]["medium"]), float(ncfg["lrs"]["slow"])
    use_ema = bool(ncfg["ema"]["enabled"])
    if method in ("single_rate", "ewc", "full_online", "nested_two_level"):
        medium = slow = 1 if method == "full_online" else matched_period(codec, ncfg)
    # the methods differ in *when* a level reaches the receiver, never in how it learns
    plan = [("rx_bias", medium, lr_medium, False)]
    plan += [(slot_group(k), slow, lr_slow, use_ema) for k in range(n_slots_of(codec, method))]
    if ncfg["adapt_codebook"]:
        plan.append(("codebook", slow, float(ncfg["codebook_lr"]), False))
    # every gradient level is computed at the transmitter and must be shipped
    return [Level(name, "rx", groups[name], period, lr, ema) for name, period, lr, ema in plan]


class SlotSelector:
    """Transmitter-side choice of the receiver's active LoRA slot.

    Given the cross-entropy of the current text under the frozen prior
    (``None``) and under every slot, it keeps the current choice unless
    another one is better by ``margin``. When the frozen prior itself is the
    best the receiver has, a slot that was never trained is started instead,
    so that a new domain gets an adapter of its own while the adapters of
    earlier domains stay intact."""

    def __init__(self, n_slots: int, margin: float):
        self.n_slots, self.margin = n_slots, float(margin)
        self.active: int | None = 0
        self.trained = [False] * n_slots
        self.bits = max(1, math.ceil(math.log2(n_slots + 1)))  # cost of signalling one choice
        self.switches = 0

    def choose(self, scores: dict) -> int | None:
        best = min(scores, key=scores.get)
        chosen = best if scores[best] < scores[self.active] - self.margin else self.active
        if chosen is None:
            fresh = [k for k, used in enumerate(self.trained) if not used]
            if fresh:
                chosen = fresh[0]
        self.switches += int(chosen != self.active)
        self.active = chosen
        return chosen

    def mark_trained(self) -> None:
        if self.active is not None:
            self.trained[self.active] = True


def make_cache(codec: SemanticCodec, cfg, method: str) -> TokenCache | None:
    ccfg = cfg["cache"]
    if method in NO_CONTEXT or not ccfg["enabled"]:
        return None
    return TokenCache(codec.vocab, float(ccfg["decay"]), float(ccfg["min_mass"]), codec.device)


def estimate_fisher(codec: SemanticCodec, params, texts: list[str], m: int, snr_db: float, n_batches: int, batch_size: int, seed: int):
    """Diagonal empirical Fisher of the base task (for the EWC baseline),
    normalised to unit mean so that ``ewc_lambda`` has a scale-free meaning."""
    rng = np.random.default_rng(seed)
    channel = Channel("awgn", seed=seed + 7)
    fisher = [torch.zeros_like(p) for p in params]
    for _ in range(n_batches):
        idx = rng.integers(0, len(texts), size=batch_size)
        batch = codec.tokenize([texts[int(i)] for i in idx])
        y, prec = codec.transmit(codec.encode(batch), batch.lens, snr_db, channel)
        ce, _, _ = codec.loss_at(batch, y, prec, m, codec.prior_tf(batch))
        grads = torch.autograd.grad(ce.mean(), params, allow_unused=True)
        codec.invalidate_codebook()
        for f, g in zip(fisher, grads):
            if g is not None:
                f.add_(g.detach() ** 2 / n_batches)
    mean = sum(float(f.sum()) for f in fisher) / max(sum(f.numel() for f in fisher), 1)
    if mean > 0:
        for f in fisher:
            f.div_(mean)
    return fisher


def run_stream(cfg, method: str, seed: int, ckpt_dir: str | Path, out_path: str | Path) -> dict:
    set_seed(seed)
    device = get_device(cfg["device"])
    scfg, ncfg = cfg["stream"], cfg["nested"]
    codec = load_codec(cfg, ckpt_dir, device)
    codec.freeze_all()
    codec.eval()

    names = list(scfg["domains"]) if scfg["kind"] == "hier" else [scfg["temporal_corpus"]]
    corpora = {name: get_corpus(name, cfg["data"]) for name in names}
    stream = build_stream(corpora, scfg, seed)
    sim = load_similarity(cfg["eval"]["sim_model"], device)

    levels = build_levels(codec, method, ncfg)
    for level in levels:
        for _, param in level.params:
            param.requires_grad_(True)
    optimizer = NestedOptimizer(
        levels,
        sync_bits=int(ncfg["sync"]["bits"]),
        sync_topk=float(ncfg["sync"]["topk"]),
        ema_decay=float(ncfg["ema"]["decay"]),
        ema_beta=float(ncfg["ema"]["beta"]),
        grad_clip=float(ncfg["grad_clip"]),
        rule=str(ncfg["rule"]),
        on_change=codec.invalidate_codebook,
    )
    trainable = optimizer.trainable()
    level_names = {lv.name for lv in levels}
    has_lora = "rx_lora" in level_names
    codec.train_codebook = "codebook" in level_names
    n_slots = n_slots_of(codec, method) if has_lora else 0
    selector = SlotSelector(n_slots, float(ncfg["select_margin"])) if n_slots > 1 else None
    select_every = int(ncfg["select_every"])
    codec.set_slot(0)

    # fast memory: per conversation, what the receiver decoded of the current
    # document (and, for training, the transmitter's mirror with the true tokens)
    use_context = bool(cfg["context"]["enabled"]) and method not in NO_CONTEXT
    max_context = int(cfg["context"]["max_tokens"])
    n_talks = int(scfg["batch_size"])
    ctx_rx: list[list[list[int]]] = [[] for _ in range(n_talks)]  # per conversation: the decoded sentences so far
    ctx_tx: list[list[list[int]]] = [[] for _ in range(n_talks)]
    cache = make_cache(codec, cfg, method)  # optional token cache of the receiver (its own decisions)
    tx_cache = make_cache(codec, cfg, method) if levels else None  # the transmitter's mirror (true tokens)

    m_op = int(scfg["m_op"])
    k_op = codec.grans.index(m_op)
    sizes = torch.tensor(codec.grans, dtype=torch.float32, device=device)
    thresholds = [float(t) for t in scfg["thresholds"]]
    tau_op = float(scfg["threshold"])
    snr_mean = float(scfg["snr"]["mean"])
    eval_batch = int(cfg["eval"]["batch_size"])
    channel = Channel(cfg["channel"]["type"], seed=seed * 1000 + 1)
    train_channel = Channel(cfg["channel"]["type"], seed=seed * 1000 + 2)  # the transmitter's own simulation
    probe_channel = Channel(cfg["channel"]["type"], seed=777)
    rng = np.random.default_rng(seed + 99)

    # identical probes for every method and seed; a probe sentence comes with the
    # part of its document that precedes it, which context-using methods may read
    probes = {}
    for name, items in build_probes(corpora, scfg, seed=0).items():
        probes[name] = {
            "texts": [p.text for p in items],
            "continued": [bool(p.history) for p in items] if use_context else None,
            "contexts": [codec.history_ids(p.history, max_context) for p in items] if use_context else None,
        }

    fisher, anchor = None, None
    if method == "ewc" and trainable:
        base_texts = [t for name in cfg["data"]["base"] for t in get_corpus(name, cfg["data"]).texts("train")]
        fisher = estimate_fisher(codec, trainable, base_texts, m_op, snr_mean, int(ncfg["ewc_batches"]), n_talks, seed)
        anchor = [p.detach().clone() for p in trainable]

    @torch.no_grad()
    def slot_scores(batch, contexts) -> dict:
        """Cross-entropy (bits/token) of a batch under the frozen prior and
        under the receiver's copy of every slot. Must run in the receiver view."""
        ids = batch.ids[batch.mask].unsqueeze(1)
        scores: dict = {}
        for candidate in [None] + list(range(n_slots)):
            if candidate is not None and not selector.trained[candidate]:
                scores[candidate] = scores[None]  # never trained: identical to the frozen prior
                continue
            codec.set_slot(candidate)
            log_prior = codec._adjust_prior(codec.lm_logits_tf(batch, contexts, flat=True), None)
            scores[candidate] = -log_prior.gather(1, ids).mean().item() / LN2
        codec.set_slot(selector.active)
        return scores

    def probe(step: int, regime: int) -> dict:
        """Held-out sentences of every slow regime, decoded by the receiver
        with the memories it holds at this moment. With several slots, the
        slot is chosen for each probe set as it would be if that regime
        returned."""
        row = {"step": step, "regime": regime, "domains": {}, "staleness": optimizer.staleness(step)}
        was_training, codec.train_codebook = codec.train_codebook, False
        with optimizer.receiver_view():
            codec.invalidate_codebook()
            for name, data in probes.items():
                texts, flags, contexts = data["texts"], data["continued"], data["contexts"]
                if selector is not None:
                    head = slice(0, eval_batch)
                    scores = slot_scores(codec.tokenize(texts[head], flags[head] if flags else None), contexts[head] if contexts else None)
                    codec.set_slot(min(scores, key=scores.get))
                probe_channel.reseed(777)
                entry = evaluate_tf(codec, texts, m_op, snr_mean, probe_channel, batch_size=eval_batch, contexts=contexts, continued=flags)
                probe_channel.reseed(778)
                few = slice(0, int(scfg["gen_sentences"]) * 4)
                entry.update(evaluate_gen(codec, texts[few], m_op, snr_mean, probe_channel, sim, batch_size=eval_batch, contexts=contexts[few] if contexts else None, continued=flags[few] if flags else None))
                entry["slot"] = -1 if codec.slot is None else int(codec.slot)
                row["domains"][name] = entry
            codec.set_slot(selector.active if selector is not None else 0)
        codec.train_codebook = was_training
        codec.invalidate_codebook()
        return row

    timer = Timer()
    log = {k: [] for k in ("ce_bits", "tok_acc", "prior_bits", "acc_by_m", "ce_by_m", "sel_symbols", "sel_acc", "sel_fb_bits", "sel_rounds", "slot", "snr_db", "codebook_power", "domain", "topic", "regime")}
    # selective refinement for every first shell and threshold: running sums of
    # (symbols, token accuracy, bitmap feedback bits, rounds per message) and the feedback decisions
    n_levels = len(codec.grans)
    grid_sum = np.zeros((n_levels, len(thresholds), 4))
    grid_meter = [[FeedbackMeter(n_levels) for _ in thresholds] for _ in range(n_levels)]
    live_every = int(scfg["live_every"])
    live_thresholds = [float(t) for t in (scfg["live_thresholds"] or thresholds)]  # the thresholds that are also decoded for real
    live_log: list[dict] = []
    live_meter = [FeedbackMeter(n_levels) for _ in live_thresholds]
    conf_all, correct_all = [], []  # for the calibration error of the receiver's confidence over the stream
    gen_log, probe_log = [], [probe(0, -1)]
    n_steps, n_tokens, n_messages = len(stream), 0, 0
    print(
        f"[stream] method={method} seed={seed} steps={n_steps} device={device} context={'on' if use_context else 'off'} "
        f"slots={n_slots} levels=" + ", ".join(f"{lv.name}@{lv.period}" for lv in levels)
    )

    for item in stream:
        for talk, start in enumerate(item.starts):  # a new document: the conversation so far is no longer context
            if start:
                ctx_rx[talk], ctx_tx[talk] = [], []
        continued = [not start for start in item.starts] if use_context else None
        batch = codec.tokenize(item.texts, continued)
        rx_contexts = [join_context(c, max_context) for c in ctx_rx] if use_context else None
        tx_contexts = [join_context(c, max_context) for c in ctx_tx] if use_context else None

        if selector is not None and (item.t - 1) % select_every == 0:  # which slot serves this text best?
            with optimizer.receiver_view():
                selector.choose(slot_scores(batch, tx_contexts))
            optimizer.ledger.add(item.t, "slot_select", selector.bits)
        active = selector.active if selector is not None else (0 if has_lora else None)
        codec.set_slot(0 if active is None and not has_lora else active)
        train_lora = has_lora and active is not None
        # a second LLM pass is needed when the transmitter's view differs from the receiver's:
        # its working adapter has moved on, or the receiver's context contains decoding errors
        second_pass = bool(levels) and ((train_lora and optimizer.is_dirty(slot_group(active))) or rx_contexts != tx_contexts)

        with optimizer.receiver_view():  # ---- what the receiver does: scored before any update
            if int(scfg["gen_every"]) > 0 and item.t % int(scfg["gen_every"]) == 0:
                few = slice(0, int(scfg["gen_sentences"]))
                gen = evaluate_gen(codec, item.texts[few], m_op, item.snr_db, channel, sim, cache=cache, batch_size=eval_batch, contexts=rx_contexts[few] if use_context else None, continued=continued[few] if use_context else None)
                gen_log.append({"step": item.t, "domain": item.domain, **gen})
            with torch.no_grad():
                y, prec = codec.transmit(codec.encode(batch), batch.lens, item.snr_db, channel)
                power = codec.codebook_power(batch, m_op)
                ids, y_flat, w_flat = codec.flat_channel(batch, y, prec)  # one row per real token
            with torch.set_grad_enabled(train_lora and not second_pass):
                logits = codec.lm_logits_tf(batch, rx_contexts, flat=True)
            with torch.no_grad():
                rx_prior = codec._adjust_prior(logits, cache)
                stats = level_stats(codec, ids, y_flat, w_flat, rx_prior)  # (tokens, K) at every prefix length
                prior_ce = -rx_prior.gather(1, ids.unsqueeze(1)).mean().item() / LN2
                live_now = None
                if live_every > 0 and item.t % live_every == 0:
                    # the same received symbols, decoded for real: free-running, round by round
                    live_now, first_pass = [], None
                    truth = list(zip(batch.ids.tolist(), batch.lens.tolist()))
                    hit_rate = lambda rows: sum(int(a == b) for (row, length), got in zip(truth, rows) for a, b in zip(row[:length], got)) / max(n_now, 1)  # noqa: E731
                    n_now = int(batch.lens.sum())
                    for index, tau in enumerate(live_thresholds):
                        out = progressive_decode(codec, y, prec, batch.lens, tau, rx_contexts, cache, first_pass=first_pass)
                        first_pass = out["first_pass"]  # the first round is the same for every threshold
                        live_meter[index].merge(out["meter"])
                        live_now.append({"symbols": out["symbols"], "tok_acc": hit_rate(out["token_ids"]), "feedback_bits": out["feedback_bits"], "rounds": out["rounds"]})
                    # the fixed-prefix receiver decoded for real as well: the reference must be audited like the proposal
                    live_fixed_now = [hit_rate(first_pass.token_ids)] + [hit_rate(codec.decode(y, m, prec, batch.lens, "llm", cache, rx_contexts).token_ids) for m in codec.grans[1:]]
        # selective refinement, emulated teacher-forced: every token stops at the
        # first shell at which the receiver's own confidence reaches the threshold
        selective = {}
        for tau in thresholds:
            stop = stop_levels(stats["conf"], tau)
            selective[tau] = (sizes[stop].mean().item(), stats["correct"].gather(1, stop.unsqueeze(1)).mean().item())
        decided = stats["decided"].gather(1, stop_levels(stats["conf"], tau_op).unsqueeze(1)).squeeze(1)
        # the same policy for every choice of the first shell, with the return-link signalling it needs
        lengths = batch.lens.tolist()
        for first in range(n_levels):
            for index, tau in enumerate(thresholds):
                stop = stop_levels(stats["conf"], tau, first)
                grid_sum[first, index] += (
                    sizes[stop].mean().item(),
                    stats["correct"].gather(1, stop.unsqueeze(1)).mean().item(),
                    feedback_bits_per_token(stop, n_levels, first).mean().item(),
                    rounds_per_message(stop, lengths, first),
                )
                grid_meter[first][index].add_stops(stop, first)
                if first == 0:
                    selective[tau] = selective[tau] + (feedback_bits_per_token(stop, n_levels).mean().item(), rounds_per_message(stop, lengths))
        if live_now is not None:
            live_log.append({"step": item.t, "live": live_now, "emulated": [{"symbols": selective[tau][0], "tok_acc": selective[tau][1]} for tau in live_thresholds], "live_fixed": live_fixed_now, "emulated_fixed": stats["correct"].mean(0).tolist()})

        if levels:  # ---- what the transmitter does: learn from the true text
            if codec.train_codebook:  # its own simulation of the channel, through the working codebook
                y, prec = codec.transmit(codec.encode(batch), batch.lens, item.snr_db, train_channel)
                _, y_flat, w_flat = codec.flat_channel(batch, y, prec)
            if second_pass:
                with torch.set_grad_enabled(train_lora):
                    logits = codec.lm_logits_tf(batch, tx_contexts, flat=True)
            log_prior = codec._adjust_prior(logits, tx_cache)
            # the prior must serve every prefix length (rate nesting): the reference one and a few others
            grans = [m_op] + sample_grans([m for m in codec.grans if m != m_op], int(scfg["extra_grans"]), rng)
            loss = sum(codec.flat_loss(ids, y_flat, w_flat, m, log_prior) for m in grans) / len(grans)
            if fisher is not None:
                penalty = sum((f * (p - a) ** 2).sum() for f, p, a in zip(fisher, trainable, anchor))
                loss = loss + 0.5 * float(ncfg["ewc_lambda"]) * penalty
            loss.backward()
            optimizer.step(item.t)
            codec.invalidate_codebook()
            if selector is not None and train_lora:
                selector.mark_trained()

        # ---- the memories that are fed by what was just sent
        if use_context:
            for talk, (got, sent) in enumerate(zip(decided.split(batch.lens.tolist()), ids.split(batch.lens.tolist()))):
                ctx_rx[talk].append(got.tolist())
                ctx_tx[talk].append(sent.tolist())
        if cache is not None:
            cache.update(decided)
        if tx_cache is not None:
            tx_cache.update(ids)

        acc_by_m = stats["correct"].mean(0).tolist()
        ce_by_m = (stats["ce"].mean(0) / LN2).tolist()
        log["ce_bits"].append(ce_by_m[k_op])
        log["tok_acc"].append(acc_by_m[k_op])
        log["prior_bits"].append(prior_ce)
        log["acc_by_m"].append(acc_by_m)
        log["ce_by_m"].append(ce_by_m)
        log["sel_symbols"].append([selective[tau][0] for tau in thresholds])
        log["sel_acc"].append([selective[tau][1] for tau in thresholds])
        log["sel_fb_bits"].append([selective[tau][2] for tau in thresholds])
        log["sel_rounds"].append([selective[tau][3] for tau in thresholds])
        log["slot"].append(-1 if active is None else int(active))
        conf_all.append(stats["conf"][:, k_op].cpu().numpy())
        correct_all.append(stats["correct"][:, k_op].cpu().numpy())
        log["snr_db"].append(item.snr_db)
        log["codebook_power"].append(power)
        log["domain"].append(item.domain)
        log["topic"].append(item.topic)
        log["regime"].append(item.regime)
        n_tokens += int(batch.lens.sum())
        n_messages += len(item.texts)
        if item.regime_end:
            probe_log.append(probe(item.t, item.regime))
        if item.t % 50 == 0 or item.t == n_steps:
            recent = slice(-50, None)
            print(
                f"[stream] {method} step {item.t:>5}/{n_steps} domain={item.domain:<10} tok_acc(50)={np.mean(log['tok_acc'][recent]):.3f} "
                f"prior_bits(50)={np.mean(log['prior_bits'][recent]):.2f} slot={log['slot'][-1]} sync_bits={optimizer.ledger.total_bits:,} it/s={item.t / max(timer.elapsed(), 1e-9):.2f}",
                flush=True,
            )

    final = probe_log[-1]["domains"]
    sync_bits = optimizer.ledger.total_bits
    tokens_per_message = n_tokens / max(n_messages, 1)
    payload = m_op * tokens_per_message
    traffic = float(cfg["link"]["traffic_per_step"])
    fixed_curve = [{"m": m, "tok_acc": a, "ce_bits": c} for m, a, c in zip(codec.grans, np.mean(log["acc_by_m"], axis=0).tolist(), np.mean(log["ce_by_m"], axis=0).tolist())]
    selective_curve = [{"threshold": tau, "symbols": s, "tok_acc": a} for tau, s, a in zip(thresholds, np.mean(log["sel_symbols"], axis=0).tolist(), np.mean(log["sel_acc"], axis=0).tolist())]
    target = float(scfg["target_acc"])

    # Return-link signalling of selective refinement. Request bits are converted to real
    # symbols at the capacity of the mean SNR - the same optimistic assumption under which
    # synchronisation bits are charged - so "total" is forward symbols plus signalling.
    capacity = capacity_real(snr_mean)

    def curve_cost(points: list[dict]) -> dict:
        """Symbols per token at the target accuracy: forward only, and with the
        signalling charged as an explicit bitmap or at its entropy."""
        accuracy = [p["tok_acc"] for p in points]
        return {
            "symbols_at_target": _finite(symbols_at_accuracy([p["symbols"] for p in points], accuracy, target)),
            "symbols_at_target_fb": _finite(symbols_at_accuracy([p["symbols_total"] for p in points], accuracy, target)),
            "symbols_at_target_fb_entropy": _finite(symbols_at_accuracy([p["symbols_total_entropy"] for p in points], accuracy, target)),
        }

    def with_totals(point: dict) -> dict:
        point["symbols_total"] = point["symbols"] + point["feedback_bits"] / capacity
        point["symbols_total_entropy"] = point["symbols"] + point["feedback_entropy_bits"] / capacity
        return point

    refine_grid = []
    for first in range(n_levels):
        points = []
        for index, tau in enumerate(thresholds):
            symbols, accuracy, bitmap, rounds = (grid_sum[first, index] / max(n_steps, 1)).tolist()
            points.append(with_totals({"threshold": tau, "symbols": symbols, "tok_acc": accuracy, "feedback_bits": bitmap, "feedback_entropy_bits": grid_meter[first][index].entropy_bits(), "rounds": rounds}))
        refine_grid.append({"first_m": codec.grans[first], "points": points, **curve_cost(points)})
    for point, full in zip(selective_curve, refine_grid[0]["points"]):  # the policy of the paper: first shell = shortest prefix
        point.update({key: full[key] for key in ("feedback_bits", "feedback_entropy_bits", "rounds", "symbols_total", "symbols_total_entropy")})
    operating = selective_curve[thresholds.index(tau_op)]

    # free-running audit: the real and the emulated operating curve on the same steps and symbols
    live_curve, live_fixed_curve, live_cost, emulated_cost = [], [], {}, {}
    if live_log:
        for index, tau in enumerate(live_thresholds):
            live = {key: float(np.mean([row["live"][index][key] for row in live_log])) for key in ("symbols", "tok_acc", "feedback_bits", "rounds")}
            live["feedback_entropy_bits"] = live_meter[index].entropy_bits()
            emulated = {key: float(np.mean([row["emulated"][index][key] for row in live_log])) for key in ("symbols", "tok_acc")}
            live_curve.append({"threshold": tau, **with_totals(live), "emulated_symbols": emulated["symbols"], "emulated_tok_acc": emulated["tok_acc"]})
        live_cost = curve_cost(live_curve)
        emulated_cost = {"symbols_at_target": _finite(symbols_at_accuracy([p["emulated_symbols"] for p in live_curve], [p["emulated_tok_acc"] for p in live_curve], target))}
        live_fixed_curve = [{"m": m, "tok_acc": float(np.mean([row["live_fixed"][k] for row in live_log])), "emulated_tok_acc": float(np.mean([row["emulated_fixed"][k] for row in live_log]))} for k, m in enumerate(codec.grans)]
        live_cost["fixed"] = _finite(symbols_at_accuracy([float(p["m"]) for p in live_fixed_curve], [p["tok_acc"] for p in live_fixed_curve], target))
        emulated_cost["fixed"] = _finite(symbols_at_accuracy([float(p["m"]) for p in live_fixed_curve], [p["emulated_tok_acc"] for p in live_fixed_curve], target))
    summary = {
        "mean_tok_acc": float(np.mean(log["tok_acc"])),
        "mean_ce_bits": float(np.mean(log["ce_bits"])),
        "mean_prior_bits": float(np.mean(log["prior_bits"])),
        "ece": expected_calibration_error(np.concatenate(conf_all), np.concatenate(correct_all)),
        "sel_symbols": operating["symbols"],
        "sel_tok_acc": operating["tok_acc"],
        # symbols per token needed for the target accuracy, read off each measured curve
        # (None when the curve never reaches the target)
        "symbols_at_target_selective": _finite(symbols_at_accuracy([p["symbols"] for p in selective_curve], [p["tok_acc"] for p in selective_curve], target)),
        "symbols_at_target_fixed": _finite(symbols_at_accuracy([float(p["m"]) for p in fixed_curve], [p["tok_acc"] for p in fixed_curve], target)),
        # the same with the return-link signalling of refinement charged (bitmap / entropy of the requests)
        "symbols_at_target_selective_fb": refine_grid[0]["symbols_at_target_fb"],
        "symbols_at_target_selective_fb_entropy": refine_grid[0]["symbols_at_target_fb_entropy"],
        "sel_feedback_bits": operating["feedback_bits"],
        "sel_feedback_entropy_bits": operating["feedback_entropy_bits"],
        "sel_rounds": operating["rounds"],
        "feedback_capacity": capacity,
        # free-running audit (None when stream.live_every is 0)
        "live_steps": len(live_log),
        "live_symbols_at_target": live_cost.get("symbols_at_target"),
        "live_symbols_at_target_fb": live_cost.get("symbols_at_target_fb"),
        "live_symbols_at_target_fb_entropy": live_cost.get("symbols_at_target_fb_entropy"),
        "live_emulated_symbols_at_target": emulated_cost.get("symbols_at_target"),
        "live_symbols_at_target_fixed": live_cost.get("fixed"),
        "live_emulated_symbols_at_target_fixed": emulated_cost.get("fixed"),
        "live_sel_symbols": live_curve[live_thresholds.index(tau_op)]["symbols"] if live_curve else None,
        "live_sel_tok_acc": live_curve[live_thresholds.index(tau_op)]["tok_acc"] if live_curve else None,
        "live_emulated_sel_symbols": live_curve[live_thresholds.index(tau_op)]["emulated_symbols"] if live_curve else None,
        "live_emulated_sel_tok_acc": live_curve[live_thresholds.index(tau_op)]["emulated_tok_acc"] if live_curve else None,
        "mean_codebook_power": float(np.mean(log["codebook_power"])),
        "final_probe_tok_acc": float(np.mean([v["tok_acc"] for v in final.values()])),
        "final_probe_prior_bits": float(np.mean([v["prior_bits"] for v in final.values()])),
        "final_probe_bleu1": float(np.mean([v["bleu1"] for v in final.values()])),
        "final_probe_bleu4": float(np.mean([v["bleu4"] for v in final.values()])),
        "tokens_per_message": tokens_per_message,
        "payload_symbols": payload,
        "sync_bits": sync_bits,
        "sync_bits_per_step": sync_bits / n_steps,
        "net_rate": net_semantic_rate(payload, sync_bits, n_steps, traffic, snr_mean),
        "sel_net_rate": net_semantic_rate(operating["symbols"] * tokens_per_message, sync_bits, n_steps, traffic, snr_mean),
        "seconds": timer.elapsed(),
    }
    result = {
        "method": method,
        "seed": seed,
        "stream": scfg["kind"],
        "m_op": m_op,
        "n_steps": n_steps,
        "context": use_context,
        "cache": cache is not None,
        "slots": n_slots,
        "slot_switches": selector.switches if selector is not None else 0,
        "rule": optimizer.rule,
        "threshold": tau_op,
        "target_acc": target,
        "fixed_curve": fixed_curve,
        "selective_curve": selective_curve,
        "refine_grid": refine_grid,
        "live_every": live_every,
        "live_curve": live_curve,
        "live_fixed_curve": live_fixed_curve,
        "live": live_log,
        "levels": [
            {"name": lv.name, "side": lv.side, "period": lv.period, "lr": lv.lr, "ema": lv.ema, "n_params": lv.n_params, "bits_per_update": optimizer.expected_bits_per_update(lv)}
            for lv in levels
        ],
        "summary": summary,
        "log": log,
        "gen": gen_log,
        "probes": probe_log,
        "ledger": {**optimizer.ledger.summary(), "cumulative": optimizer.ledger.cumulative(n_steps)},
        "config": cfg.to_dict(),
        "env": environment(),
    }
    save_json(out_path, result)
    print(f"[stream] {method} seed={seed}: mean tok_acc={summary['mean_tok_acc']:.4f} final-probe tok_acc={summary['final_probe_tok_acc']:.4f} sync={sync_bits:,} bits -> {out_path}")
    return result
