"""Uncertainty-triggered selective refinement.

Every token is first sent with the shortest codeword prefix. The receiver's
Bayesian decoder yields a posterior for every token; it asks for the next
shell of symbols only for the tokens it is not confident about, and repeats.
Because the shells are nested, the extra symbols simply extend what was
already received: semantic incremental redundancy, spent where the language
model cannot fill the gap on its own.

The receiver's confidence is *calibrated* on held-out data, so the stopping
threshold means something (a target probability that the token is right),
and the calibration error is reported before and after.

Refinement is not free on the return link: after every round the receiver
has to say which tokens it wants more of. That signalling is counted here
(:class:`FeedbackMeter`) in two forms - an explicit bitmap, one bit per token
still open, and the entropy of those decisions, which is what an ideal
entropy coder of the bitmap would need - and it is reported next to the
forward symbols wherever a refinement result is reported.

Two evaluations are provided:

* a teacher-forced sweep of the whole rate-accuracy curve, which costs one LLM
  pass per sentence because the prior does not depend on the prefix length;
* a free-running multi-round decode at chosen thresholds, in which the
  receiver really re-decodes after each round of requests.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .channel import Channel, shell_slices
from .codec import SemanticCodec
from .metrics import expected_calibration_error, fidelity


@torch.no_grad()
def level_stats(codec: SemanticCodec, ids: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, log_prior: torch.Tensor, entropy: bool = False) -> dict[str, torch.Tensor]:
    """Teacher-forced statistics of every token at every granularity, each of
    shape (N, K): ``correct`` (0/1), ``conf`` (posterior probability of the
    decision), ``ce`` (nats) and ``decided`` (token id); optionally the
    posterior ``entropy``.

    Inputs are flat (one row per non-padding token, see
    ``SemanticCodec.flat_channel``): ids (N,), received symbols and noise
    precisions (N, M), log-prior (N, V). The prior does not depend on the
    prefix length, so this costs one LLM pass in total."""
    rows: dict[str, list[torch.Tensor]] = {"correct": [], "conf": [], "ce": [], "decided": []}
    if entropy:
        rows["entropy"] = []
    index = ids.unsqueeze(1)
    code = codec.codebook().to(y.dtype)
    scale = float(codec.lik_weight)
    score = log_prior.clone()  # unnormalised log-posterior, built up shell by shell in place
    buffer = torch.empty_like(score)
    for sl in shell_slices(codec.grans):
        # log-likelihood of this shell (see SemanticCodec.loglik_w), added to what the shorter prefix gave
        shell, w = code[:, sl], weight[:, sl]
        score.addmm_(torch.cat([w * y[:, sl], -0.5 * w], dim=1), torch.cat([shell, shell * shell], dim=1).t(), alpha=scale)
        torch.log_softmax(score, dim=1, out=buffer)  # the normalised log-posterior at this prefix length
        best, decided = buffer.max(dim=1)
        rows["correct"].append((decided == ids).float())
        rows["conf"].append(best.exp())
        rows["ce"].append(-buffer.gather(1, index).squeeze(1))
        rows["decided"].append(decided)
        if entropy:
            rows["entropy"].append(-(buffer.exp() * buffer).sum(-1))
    return {key: torch.stack(value, dim=1) for key, value in rows.items()}


def stop_levels(conf: torch.Tensor, threshold: float, start: int = 0) -> torch.Tensor:
    """Index of the first shell, from shell ``start`` on, at which the
    confidence (N, K) reaches the threshold; the last shell if it never does.
    ``start`` is the shell sent in the first round (0: the shortest prefix)."""
    hit = conf >= threshold
    if start > 0:
        hit = hit.clone()
        hit[:, :start] = False
    last = conf.shape[1] - 1
    return torch.where(hit.any(dim=1), hit.float().argmax(dim=1), torch.full_like(hit[:, 0], last, dtype=torch.long))


def binary_entropy(p: float) -> float:
    """Entropy of a Bernoulli(p) decision in bits."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


class FeedbackMeter:
    """Return-link signalling of selective refinement.

    Protocol: round ``r`` delivers shell ``r`` of every token that is still
    open. After the round the receiver tells the transmitter, for each open
    token, whether it wants the next shell; a token that is not continued is
    closed for good, and after the last shell nothing has to be said. With
    ``K`` shells there are therefore at most ``K - 1`` decisions per token.

    Two costs are derived from the decisions that were actually taken:

    ``bitmap_bits``   one bit per open token and round (an explicit bitmap);
    ``entropy_bits``  the empirical entropy of the decisions of every round,
                      i.e. what an ideal entropy coder of the bitmap that
                      knows the request rate of each round would spend. It is
                      a lower reference, not an implemented code.

    Both are bits per transmitted token."""

    def __init__(self, n_levels: int):
        self.n_levels = int(n_levels)
        self.open = np.zeros(max(self.n_levels - 1, 0), dtype=np.float64)  # tokens deciding after round r
        self.cont = np.zeros(max(self.n_levels - 1, 0), dtype=np.float64)  # of which: continue
        self.tokens = 0.0

    def add_round(self, round_index: int, n_open: int, n_continue: int) -> None:
        """Decisions taken after round ``round_index``. With reopening a
        decoder can take more rounds than there are shells; the account grows."""
        if round_index >= len(self.open):
            grow = round_index + 1 - len(self.open)
            self.open = np.concatenate([self.open, np.zeros(grow)])
            self.cont = np.concatenate([self.cont, np.zeros(grow)])
        self.open[round_index] += n_open
        self.cont[round_index] += n_continue

    def add_tokens(self, n: int) -> None:
        self.tokens += n

    def add_stops(self, stop, start: int = 0) -> None:
        """Account for tokens that stop at shell index ``stop`` (array-like)
        when the first round sends shell ``start``."""
        stop = np.asarray(stop.detach().cpu() if isinstance(stop, torch.Tensor) else stop).reshape(-1)
        self.tokens += stop.size
        for r in range(start, self.n_levels - 1):
            self.open[r] += float((stop >= r).sum())
            self.cont[r] += float((stop > r).sum())

    def bitmap_bits(self) -> float:
        return float(self.open.sum() / self.tokens) if self.tokens else 0.0

    def entropy_bits(self) -> float:
        if not self.tokens:
            return 0.0
        total = sum(n * binary_entropy(c / n) for n, c in zip(self.open, self.cont) if n > 0)
        return float(total / self.tokens)

    def merge(self, other: "FeedbackMeter") -> None:
        size = max(len(self.open), len(other.open))
        pad = lambda a: np.concatenate([a, np.zeros(size - len(a))])  # noqa: E731
        self.open, self.cont = pad(self.open) + pad(other.open), pad(self.cont) + pad(other.cont)
        self.tokens += other.tokens

    def summary(self) -> dict[str, float]:
        return {"feedback_bits": self.bitmap_bits(), "feedback_entropy_bits": self.entropy_bits()}


def feedback_bits_per_token(stop: torch.Tensor, n_levels: int, start: int = 0) -> torch.Tensor:
    """Bitmap feedback of every token (float tensor like ``stop``): one
    decision after each round it takes part in, none after the last shell."""
    return ((stop + 1).clamp(max=n_levels - 1) - start).clamp(min=0).to(torch.float32)


def rounds_per_message(stop: torch.Tensor, lengths: list[int], start: int = 0) -> float:
    """Mean number of transmission rounds per message when every token stops
    at shell index ``stop`` (flat, one entry per token; ``lengths`` cuts it
    into messages)."""
    if not lengths:
        return 0.0
    return float(np.mean([int(part.max()) - start + 1 for part in stop.split(lengths)]))


@torch.no_grad()
def progressive_decode(codec: SemanticCodec, y: torch.Tensor, prec: torch.Tensor, lens: torch.Tensor, threshold: float, contexts: list[list[int]] | None = None, cache=None, start: int = 0, first_pass=None, beam: int = 1, reopen: bool = False) -> dict:
    """Selective refinement decoded for real (free-running) on symbols that
    were transmitted once with :meth:`SemanticCodec.transmit`.

    The receiver first reads shell ``start`` of every token and decodes the
    sentence with its own earlier decisions as the language-model input. A
    token whose posterior confidence reaches ``threshold`` is closed; for the
    others the next shell is read and the sentence is decoded again, until
    every token is closed or has all its shells. This is exactly the policy
    that the stream runner emulates teacher-forced, on the same received
    symbols, so the difference between the two is the effect of free-running
    decoding and nothing else.

    A message without open tokens is finished and is not decoded again. The
    first round does not depend on the threshold: its result is returned as
    ``first_pass`` and may be handed to further calls on the same symbols.

    Options (the decoder of the paper is fixed in ``docs/METHOD.md``):
    ``beam``    width of the beam search of every decoding pass (1: greedy);
    ``reopen``  a token closed in an earlier round may be requested again
                when re-decoding with a corrected context lowers its
                confidence below the threshold. Every token that still has
                shells left then takes part in every round's decisions, and
                the signalling is charged for all of them.

    Returns the final decisions (``token_ids``), the shell reached by every
    token (``level``), the mean number of symbols per token, the feedback
    account and the mean number of rounds per message."""
    grans = codec.grans
    n_levels = len(grans)
    device = y.device
    sizes = torch.tensor(grans, dtype=torch.long, device=device)
    bsz, width, _ = y.shape
    lens = lens.to(device)
    mask = torch.arange(width, device=device).unsqueeze(0) < lens.unsqueeze(1)
    level = torch.full((bsz, width), int(start), dtype=torch.long, device=device)
    is_open = mask.clone()
    rounds = torch.ones(bsz, dtype=torch.long, device=device)
    meter = FeedbackMeter(n_levels)
    meter.add_tokens(int(mask.sum()))
    token_ids: list[list[int]] = [[] for _ in range(bsz)]
    active = torch.arange(bsz, device=device)  # messages that are still being refined
    round_index = start
    limit = start + (n_levels - start) * max(width, 1)  # with reopening every round still raises at least one token
    while True:
        if round_index == start and first_pass is not None:
            decoded = first_pass
        else:
            span = int(lens[active].max())
            picked = active.tolist()
            decoded = codec.decode(y[active, :span], sizes[level[active, :span]], prec[active], lens[active], "llm", cache, [contexts[i] for i in picked] if contexts else None, beam=beam)
        if round_index == start:
            first_pass = decoded
        span = decoded.token_logprob.shape[1]
        for row, got in zip(active.tolist(), decoded.token_ids):
            token_ids[row] = got
        can = mask[active, :span] & (level[active, :span] < n_levels - 1)  # tokens that still have shells left
        is_open[active, :span] = can if reopen else (is_open[active, :span] & can)
        if not bool(is_open[active].any()) or round_index >= limit:
            break
        conf = decoded.token_logprob.exp().to(device)
        request = is_open[active, :span] & (conf < threshold)
        meter.add_round(round_index, int(is_open[active].sum()), int(request.sum()))
        is_open[active, :span] = request
        again = request.any(dim=1)
        if not bool(again.any()):
            break
        level[active, :span] += request.long()
        active = active[again]
        rounds[active] += 1
        round_index += 1
    n_tokens = max(int(mask.sum()), 1)
    return {
        "token_ids": token_ids,
        "level": level,
        "symbols": float(sizes[level][mask].sum()) / n_tokens,
        "rounds": float(rounds.float().mean()),
        "n_tokens": int(mask.sum()),
        "meter": meter,
        "first_pass": first_pass,
        **meter.summary(),
    }


@torch.no_grad()
def collect(codec: SemanticCodec, texts: list[str], snr_db: float, channel: Channel, batch_size: int = 16) -> dict:
    """Teacher-forced token statistics at every granularity from one channel
    realisation. Returns arrays of shape (N_tokens, K): ``correct`` (0/1),
    ``conf`` (posterior probability of the decision) and ``entropy``."""
    out: dict[str, list[np.ndarray]] = {"correct": [], "conf": [], "entropy": []}
    for i in range(0, len(texts), batch_size):
        batch = codec.tokenize(texts[i : i + batch_size])
        y, prec = codec.transmit(codec.encode(batch), batch.lens, snr_db, channel)
        stats = level_stats(codec, *codec.flat_channel(batch, y, prec), codec.prior_tf(batch, flat=True), entropy=True)
        for key in out:
            out[key].append(stats[key].cpu().numpy())
    return {key: np.concatenate(value) for key, value in out.items()}


def features(conf: np.ndarray, entropy: np.ndarray, level: np.ndarray, n_levels: int) -> np.ndarray:
    """Calibrator inputs for token decisions: the logit of the decoder's own
    posterior confidence and the (scaled) index of the last shell received.
    With unit slope and zero offsets the calibrator is the identity, so it can
    only correct the posterior, not replace it."""
    clipped = np.clip(conf, 1e-6, 1 - 1e-6)
    return np.stack([np.log(clipped / (1 - clipped)), level / max(n_levels - 1, 1)], axis=-1)


def _grid_features(data: dict) -> np.ndarray:
    n, k = data["correct"].shape
    level = np.tile(np.arange(k, dtype=float), (n, 1))
    return features(data["conf"], data["entropy"], level, k).reshape(n * k, -1)


class Calibrator:
    """Logistic (Platt-style) map from decode statistics to P(token correct)."""

    def __init__(self, ridge: float = 1e-4):
        self.ridge = ridge
        self.mean = self.std = self.weights = None

    def fit(self, feats: np.ndarray, labels: np.ndarray, iters: int = 200) -> "Calibrator":
        self.mean, self.std = feats.mean(0), feats.std(0) + 1e-8
        x = np.hstack([(feats - self.mean) / self.std, np.ones((len(feats), 1))])
        w = np.zeros(x.shape[1])
        for _ in range(iters):  # Newton / IRLS with a ridge penalty
            p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
            grad = x.T @ (p - labels) / len(labels) + self.ridge * w
            hess = (x * (p * (1 - p))[:, None]).T @ x / len(labels) + self.ridge * np.eye(len(w))
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.weights = w
        return self

    def predict(self, feats: np.ndarray) -> np.ndarray:
        x = np.hstack([(feats - self.mean) / self.std, np.ones((len(feats), 1))])
        return 1.0 / (1.0 + np.exp(-np.clip(x @ self.weights, -30, 30)))


def stopping_curve(prob: np.ndarray, quality: np.ndarray, grans: list[int], thresholds: np.ndarray, gamma: float = 0.5) -> list[dict]:
    """Per item, stop at the first shell whose calibrated confidence reaches
    the threshold; otherwise use everything. ``quality`` is (N, K)."""
    n, k = prob.shape
    sizes = np.asarray(grans, dtype=float)
    rows = []
    for tau in thresholds:
        hit = prob >= tau
        stop = np.where(hit.any(1), hit.argmax(1), k - 1)
        q = quality[np.arange(n), stop]
        meter = FeedbackMeter(k)
        meter.add_stops(stop)
        rows.append({"threshold": float(tau), "symbols": float(sizes[stop].mean()), "quality": float(q.mean()), "success": float((q >= gamma).mean()), "rounds": float((stop + 1).mean()), **meter.summary()})
    return rows


def reference_policies(quality: np.ndarray, grans: list[int], gamma: float = 0.5) -> dict:
    """Fixed-rate operating points and the oracle that stops at the first
    shell that is actually good enough."""
    n, k = quality.shape
    sizes = np.asarray(grans, dtype=float)
    fixed = [{"m": int(grans[j]), "symbols": float(sizes[j]), "quality": float(quality[:, j].mean()), "success": float((quality[:, j] >= gamma).mean())} for j in range(k)]
    ok = quality >= gamma
    stop = np.where(ok.any(1), ok.argmax(1), k - 1)
    q = quality[np.arange(n), stop]
    return {"fixed": fixed, "oracle": {"symbols": float(sizes[stop].mean()), "quality": float(q.mean()), "success": float((q >= gamma).mean())}}


def random_matched(stop: np.ndarray, quality: np.ndarray, grans: list[int], gamma: float = 0.5, seed: int = 0) -> dict:
    """Same distribution of stopping points, assigned to items at random: the
    control that shows whether the confidence signal carries information."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(stop)
    q = quality[np.arange(len(stop)), shuffled]
    return {"symbols": float(np.asarray(grans, dtype=float)[shuffled].mean()), "quality": float(q.mean()), "success": float((q >= gamma).mean())}


def interpolate_fixed(fixed: list[dict], symbols: float) -> float:
    """Quality of time-sharing between fixed rates at the same mean rate."""
    xs = [f["symbols"] for f in fixed]
    ys = [f["quality"] for f in fixed]
    return float(np.interp(symbols, xs, ys))


@torch.no_grad()
def selective_decode(codec: SemanticCodec, texts: list[str], snr_db: float, channel: Channel, calibrator: Calibrator, threshold: float, sim=None, batch_size: int = 16) -> dict:
    """Free-running multi-round refinement at one threshold.

    Round 1 sends the first shell of every token. After each round the
    receiver decodes the sentence and requests the next shell for the tokens
    whose calibrated confidence is below ``threshold``. Every round is its own
    transmission: the requested symbols form one block per message, the block
    is power-normalised to exactly unit average power, and it sees a fresh
    channel realisation."""
    grans = codec.grans
    edges = [0] + list(grans)
    refs, hyps = [], []
    symbols = tokens = correct = requests = rounds_used = batches = 0
    meter = FeedbackMeter(len(grans))
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        batch = codec.tokenize(chunk)
        x = codec.encode(batch)  # (B, T, M)
        bsz, width, _ = x.shape
        dims = torch.arange(codec.M, device=x.device).view(1, 1, -1)
        lower = torch.tensor(edges[:-1], device=x.device)
        upper = torch.tensor(edges[1:], device=x.device)
        y = torch.zeros_like(x)
        weight = torch.zeros_like(x)  # noise precision of every received symbol; 0 = not received
        level = torch.full_like(batch.ids, -1)  # index of the last shell received, per token
        send = batch.mask.clone()  # tokens that receive their next shell this round
        alive = torch.ones(bsz, dtype=torch.bool, device=x.device)  # messages that still exchange feedback
        decoded = None
        for round_index in range(len(grans)):
            nxt = (level + 1).clamp(max=len(grans) - 1)
            block = send.unsqueeze(-1) & (dims >= lower[nxt].unsqueeze(-1)) & (dims < upper[nxt].unsqueeze(-1))
            count = block.sum(dim=(1, 2)).clamp_min(1).to(x.dtype)
            energy = (x.pow(2) * block).sum(dim=(1, 2)).clamp_min(1e-12)
            gain2 = count / energy  # block power control: unit average power per symbol
            eff = channel.effective_snr_db(bsz, snr_db).to(device=x.device, dtype=x.dtype)
            prec = gain2 * torch.pow(10.0, eff / 10.0)  # (B,)
            noise = channel.noise(x.shape, prec.rsqrt().cpu().view(-1, 1, 1)).to(device=x.device, dtype=x.dtype)
            y = torch.where(block, x + noise, y)
            weight = torch.where(block, prec.view(-1, 1, 1).expand_as(x), weight)
            symbols += float(block.sum())
            level = torch.where(send, nxt, level)
            decoded = codec.decode_loglik(codec.loglik_w(y, weight), batch.lens)
            rounds_used += 1
            if round_index == len(grans) - 1:
                break
            feats = features(decoded.token_logprob.exp().numpy(), decoded.token_entropy.numpy(), level.clamp(min=0).cpu().numpy().astype(float), len(grans))
            prob = torch.as_tensor(calibrator.predict(feats.reshape(-1, feats.shape[-1])).reshape(bsz, width), device=x.device)
            can = batch.mask & (level < len(grans) - 1)
            send = (prob < threshold) & can
            # return link: one decision per token that can still be refined, in every message
            # that has not yet answered "nothing more" (such a message is finished)
            meter.add_round(round_index, int((can & alive.unsqueeze(1)).sum()), int((send & alive.unsqueeze(1)).sum()))
            alive = alive & send.any(dim=1)
            if not bool(send.any()):
                break
            requests += int(send.sum())
        refs.extend(codec.reference_texts(chunk))
        hyps.extend(decoded.texts)
        tokens += int(batch.lens.sum())
        meter.add_tokens(int(batch.lens.sum()))
        batches += 1
        for row, length, got in zip(batch.ids.tolist(), batch.lens.tolist(), decoded.token_ids):
            correct += sum(int(a == b) for a, b in zip(row[:length], got))
    out = fidelity(refs, hyps, sim)
    out.update({"threshold": float(threshold), "symbols": symbols / max(tokens, 1), "gen_tok_acc": correct / max(tokens, 1), "requests_per_token": requests / max(tokens, 1), "rounds_per_batch": rounds_used / max(batches, 1), **meter.summary()})
    return out


def run_refinement(codec: SemanticCodec, val_texts: list[str], test_texts: list[str], snr_db: float, channel_kind: str, batch_size: int = 16) -> tuple[dict, Calibrator]:
    """Teacher-forced sweep of the selective-refinement curve at one channel
    condition, with the fixed-rate, oracle and random controls. Returns the
    result and the calibrator fitted on the validation tokens."""
    val = collect(codec, val_texts, snr_db, Channel(channel_kind, seed=4101), batch_size)
    test = collect(codec, test_texts, snr_db, Channel(channel_kind, seed=4102), batch_size)
    calibrator = Calibrator().fit(_grid_features(val), val["correct"].reshape(-1))
    prob = calibrator.predict(_grid_features(test)).reshape(test["correct"].shape)
    labels = test["correct"].reshape(-1)

    # Sweep the whole operating curve, from "always stop at the first shell" to
    # "never stop early". Thresholds are quantiles of the predicted
    # probabilities themselves (no labels involved).
    thresholds = np.unique(np.concatenate([np.quantile(prob.reshape(-1), np.linspace(0.0, 1.0, 21)), [np.inf]]))
    curve = stopping_curve(prob, test["correct"], codec.grans, thresholds)
    refs = reference_policies(test["correct"], codec.grans)
    for point in curve:  # gain over time-sharing between fixed rates at the same mean rate
        point["gain_over_fixed"] = point["quality"] - interpolate_fixed(refs["fixed"], point["symbols"])

    n, k = prob.shape
    mid = thresholds[(len(thresholds) - 1) // 2]
    hit = prob >= mid
    stop = np.where(hit.any(1), hit.argmax(1), k - 1)
    return {
        "snr_db": snr_db,
        "channel": channel_kind,
        "n_val_tokens": int(len(val["correct"])),
        "n_test_tokens": int(n),
        "ece_raw": expected_calibration_error(test["conf"].reshape(-1), labels),
        "ece_calibrated": expected_calibration_error(prob.reshape(-1), labels),
        "curve": curve,
        "random_matched_at_mid_threshold": {"threshold": float(mid), **random_matched(stop, test["correct"], codec.grans)},
        "calibrator": {"mean": calibrator.mean.tolist(), "std": calibrator.std.tolist(), "weights": calibrator.weights.tolist()},
        **refs,
    }, calibrator
