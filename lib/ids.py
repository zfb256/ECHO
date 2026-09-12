from __future__ import annotations

import hashlib


def stable_id(*parts: object, length: int = 16) -> str:
    text = "\u241f".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]

