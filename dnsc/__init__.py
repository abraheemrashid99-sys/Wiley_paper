"""Doubly Nested Semantic Communication (DNSC).

A rate- and time-nested LLM semantic codec for drift-resilient 6G links.

* nested in rate: the transmitted embedding is prefix-truncatable, so the
  payload (not the model) adapts to the link;
* nested in time: the transceiver's parameters are split into levels that are
  updated, and synchronised over the air, at separated frequencies.
"""

import os

# Must be set before torch is imported: lets PyTorch fall back to the CPU for
# the few operators that Apple's MPS backend does not implement.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"
