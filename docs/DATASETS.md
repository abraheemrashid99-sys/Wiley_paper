# Datasets

Four public text corpora, all small enough for a 16 GB laptop: the raw
downloads total well under 200 MB and each is capped at 60 000 sentences after
filtering. Every corpus is reduced to single sentences of 4–30 words (the
length protocol used by DeepSC and most text semantic-communication papers),
de-duplicated, and split 80/10/10 **by document**.

| Corpus | Role in the study | What it supplies | Size used | Source | Licence |
| --- | --- | --- | --- | --- | --- |
| **Europarl v7** (English side) | Base knowledge: the semantic codebook and its power normalisation are trained here; rate and refinement results are reported here | The reference corpus of text semantic communication; makes our base numbers comparable with the literature | ≤ 60 000 sentences, streamed (only the rows used are downloaded) | Hugging Face `Helsinki-NLP/europarl`, config `en-ro` | No known copyright restrictions (per the corpus authors) |
| **AG News** | Stream domain 1; medium-timescale *topic* drift | 4 topic classes (world, sports, business, sci/tech) to switch between | 60 000 sentences from 120 000 articles (≈ 30 MB) | GitHub CSV mirror of the original release; Hugging Face `fancyzhx/ag_news` as fallback | Research use; no formal licence stated by the source |
| **TeleQnA** | Stream domain 2; *domain* shift into telecom / 3GPP language | 10 000 telecom questions with explanations, 5 categories incl. standards specifications — ties the study to 6G standardisation | ≈ 25 500 sentences (13 MB) | Authors' GitHub repository `netop-team/TeleQnA` (password-protected archive, password published in their README) | MIT |
| **News Category Dataset (HuffPost)** | Natural *temporal* drift; the slow timescale is real calendar time | 209 527 dated headlines and descriptions, 2012–2022, 42 categories; the news benchmark of Wild-Time | ≤ 60 000 sentences (87 MB download) | Hugging Face `heegyu/news-category-dataset` | CC BY 4.0 |

All four loaders have been run against the real sources. With the default
4–30 word filter and the 60 000 cap they give 60 000 (Europarl), 60 000
(AG News), 25 543 (TeleQnA) and 60 000 (HuffPost, 28 Jan 2012 – 23 Sep 2022,
42 categories) sentences. `configs/laptop.yaml` uses 4–20 words and a 40 000
cap to keep CPU runs affordable.

`configs/rehearsal.yaml` needs only AG News and TeleQnA, both served from
GitHub, so it also runs where Hugging Face is unreachable. It trains its own
small language model on AG News and is a check of the code, not a result.

## Why these four

The method claims to exploit drift that is **hierarchical in time**. A fair
test therefore needs drift at more than one timescale that we did not design
to suit the method:

- **Fast** — the channel. Per-step SNR follows a clipped AR(1) process; fading
  is Rayleigh block fading. This is the only synthetic band, and it is the
  standard one.
- **Medium** — the topic of conversation. Supplied by the class labels of AG
  News and the category labels of TeleQnA.
- **Slow** — the knowledge domain (parliamentary → news → telecom) in the
  `hier` stream, and real calendar time in the `temporal` stream.

Sentences keep their document id and order (a TeleQnA document is a question,
its answer and the explanation; an AG News document is a title and its
description). The streams currently draw sentences independently within a
topic; `scripts/headroom.py` measures what the preceding sentence of the same
document would be worth as context for the LLM, which decides whether
document-coherent streams are the next step.

The `temporal` stream is the important control. HuffPost is replayed in
chronological order with no regime boundaries placed by us, so a method cannot
win there because a benchmark generator was tuned in its favour. If the
proposed method helps on `hier` but not on `temporal`, that is a finding and
will be reported as one.

## Two streams

`stream.kind=hier` — domains are visited for `tau_slow` steps each and the tour
is repeated (`cycles`), so each domain is *revisited*: retention of earlier
knowledge is measured directly on held-out probes at every regime boundary.
Within a domain the topic switches every `tau_med` steps; `kappa` is the share
of each batch drawn from the current topic.

`stream.kind=temporal` — a window slides through the date-sorted corpus; probes
are held-out sentences from each of `temporal_eras` equal-sized eras.

## Citations

- P. Koehn, "Europarl: A parallel corpus for statistical machine translation,"
  MT Summit, 2005.
- X. Zhang, J. Zhao, Y. LeCun, "Character-level convolutional networks for text
  classification," NeurIPS, 2015. (AG News)
- A. Maatouk et al., "TeleQnA: A benchmark dataset to assess large language
  models telecommunications knowledge," arXiv:2310.15051, 2023.
- R. Misra, "News Category Dataset," arXiv:2209.11429, 2022.
- H. Yao et al., "Wild-Time: A benchmark of in-the-wild distribution shift over
  time," NeurIPS, 2022.
- H. Xie, Z. Qin, G. Y. Li, B.-H. Juang, "Deep learning enabled semantic
  communication systems," IEEE Trans. Signal Process., 2021. (DeepSC protocol)

Check each reference against the published version before it goes into the
manuscript.

## Optional extensions (not wired in)

- **PTB-XL** — would connect the paper to your nested ECG encoder, but it is a
  different modality and would dilute a text-LLM paper.
- **TSpec-LLM / 3GPP specifications** — richer telecom text, but roughly 15 GB.
- **arXiv abstracts by year** — a second natural temporal stream.
