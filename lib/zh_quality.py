from __future__ import annotations

import re
from typing import Any


_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_CJK_GAP = re.compile(r"(?<=[\u3400-\u4dbf\u4e00-\u9fff])\s+(?=[\u3400-\u4dbf\u4e00-\u9fff])")
_MOBILE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_LANDLINE = re.compile(r"(?<!\d)0\d{2,3}[-—－ ]?\d{7,8}(?!\d)")
_SENSITIVE = ("身份证", "银行卡", "手机号", "家庭住址", "裸照", "自杀方法", "炸弹制作")


def normalize_zh_value(value: Any) -> Any:
    """Remove corpus tokenization spaces between Chinese characters, recursively."""
    if isinstance(value, str):
        return _CJK_GAP.sub("", value).strip()
    if isinstance(value, list):
        return [normalize_zh_value(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_zh_value(v) for k, v in value.items()}
    return value


def flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(flatten_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(flatten_text(v) for v in value.values())
    return ""


def dialogue_only(payload: dict[str, Any]) -> Any:
    """Select annotator-visible dialogue, excluding KdConv KB/attribute metadata."""
    messages = payload.get("messages")
    if isinstance(messages, dict) and isinstance(messages.get("message"), list):
        return messages["message"]
    for key in ("dialog", "dialogue", "conversation", "history", "utterances", "text"):
        if key in payload:
            return payload[key]
    return payload


def chinese_dialogue_ok(payload: dict[str, Any]) -> tuple[bool, str]:
    text = flatten_text(dialogue_only(payload))
    compact = "".join(text.split())
    if len(compact) < 20:
        return False, "too_short"
    if len(compact) > 8000:
        return False, "too_long"
    cjk = len(_CJK.findall(compact))
    if cjk / max(1, len(compact)) < 0.35:
        return False, "low_chinese_ratio"
    if any(term in text for term in _SENSITIVE):
        return False, "sensitive_term"
    if _MOBILE.search(text) or _LANDLINE.search(text):
        return False, "contact_number"
    return True, "ok"
