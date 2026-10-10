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

Two evaluations are provided:

* a teacher-forced sweep of the whole rate-accuracy curve, which costs one LLM
  pass per sentence because the prior does not depend on the prefix length;
* a free-running multi-round decode at chosen thresholds, in which the
  receiver really re-decodes after each round of requests.
"""

from __future__ import annotations

import numpy as np
import torch

from .channel import Channel
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
    for m in codec.grans:
        score = log_prior + codec.lik_weight * codec.loglik_w(y[:, :m], weight[:, :m])  # unnormalised log-posterior
        norm = torch.logsumexp(score, dim=-1)
        best, decided = score.max(dim=-1)
        rows["correct"].append((decided == ids).float())
        rows["conf"].append((best - norm).exp())
        rows["ce"].append(norm - score.gather(1, index).squeeze(1))
        rows["decided"].append(decided)
        if entropy:
            post = score - norm.unsqueeze(1)
            rows["entropy"].append(-(post.exp() * post).sum(-1))
    return {key: torch.stack(value, dim=1) for key, value in rows.items()}


def stop_levels(conf: torch.Tensor, threshold: float) -> torch.Tensor:
    """Index of the first shell at which the confidence (N, K) reaches the
    threshold; the last shell if it never does."""
    hit = conf >= threshold
    last = conf.shape[1] - 1
    return torch.where(hit.any(dim=1), hit.float().argmax(dim=1), torch.full_like(hit[:, 0], last, dtype=torch.long))


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
        rows.append({"threshold": float(tau), "symbols": float(sizes[stop].mean()), "quality": float(q.mean()), "success": float((q >= gamma).mean()), "rounds": float((stop + 1).mean())})
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
            send = (prob < threshold) & batch.mask & (level < len(grans) - 1)
            if not bool(send.any()):
                break
            requests += int(send.sum())
        refs.extend(codec.reference_texts(chunk))
        hyps.extend(decoded.texts)
        tokens += int(batch.lens.sum())
        batches += 1
        for row, length, got in zip(batch.ids.tolist(), batch.lens.tolist(), decoded.token_ids):
            correct += sum(int(a == b) for a, b in zip(row[:length], got))
    out = fidelity(refs, hyps, sim)
    out.update({"threshold": float(threshold), "symbols": symbols / max(tokens, 1), "gen_tok_acc": correct / max(tokens, 1), "requests_per_token": requests / max(tokens, 1), "rounds_per_batch": rounds_used / max(batches, 1)})
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
