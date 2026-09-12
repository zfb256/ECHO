from __future__ import annotations

import json
from typing import Any


PREFERRED_TEXT_KEYS = (
    "dialog",
    "dialogue",
    "conversation",
    "history",
    "utterances",
    "response",
    "text",
    "question",
)

# When a dialogue field is a list of turn dicts, pull the text/speaker from these keys.
TURN_TEXT_KEYS = ("text", "utterance", "content", "message", "value", "response")
TURN_SPEAKER_KEYS = ("speaker", "role", "sender", "name", "agent")


def _turns_from_list(items: list) -> list[str] | None:
    """One clean line per turn from a list of strings or {speaker,text} dicts.

    Returns None if the list is not a recognizable turn list (so the caller can fall back),
    rather than letting `_stringify` treat dictionary keys as dialogue turns.
    """
    lines: list[str] = []
    for it in items:
        if isinstance(it, str):
            s = it.strip()
            if s:
                lines.append(s)
        elif isinstance(it, dict):
            txt = next((it[k] for k in TURN_TEXT_KEYS if isinstance(it.get(k), str) and it[k].strip()), None)
            if txt is None:
                return None  # dict without a recognizable text field -> not a turn list
            spk = next((str(it[k]) for k in TURN_SPEAKER_KEYS if it.get(k) not in (None, "")), None)
            lines.append(f"{spk}: {txt.strip()}" if spk else txt.strip())
        else:
            return None
    return lines or None


def _turns_from_string(text: str) -> list[str] | None:
    """Parse string-encoded dialogues without collapsing turn boundaries."""
    stripped = text.strip()
    if not stripped:
        return None

    if stripped[0] in "[{":
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            turns = _turns_from_list(parsed)
            if turns:
                return turns

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    return lines if len(lines) >= 2 else None


def _turns_from_history_response(payload: dict[str, Any]) -> list[str] | None:
    history = payload.get("history")
    response = payload.get("response")
    if not isinstance(history, list) or not isinstance(response, str) or not response.strip():
        return None
    turns = _turns_from_list(history)
    if not turns:
        return None
    turns.append(f"Assistant: {response.strip()}")
    return turns if len(turns) >= 2 else None


def _turns_from_nested_messages(payload: dict[str, Any]) -> list[str] | None:
    """Handle KdConv's ``messages: {message: [...], attrs: [...]}`` schema."""
    messages = payload.get("messages")
    if not isinstance(messages, dict):
        return None
    turns = messages.get("message")
    return _turns_from_list(turns) if isinstance(turns, list) else None


def compact_text(value: Any, max_chars: int | None = None) -> str:
    """Whitespace-normalize a value to a single compact string.

    max_chars=None means no length cap. Apply any cap after turn truncation.
    """
    text = _stringify(value)
    text = " ".join(text.split())
    return text if max_chars is None else text[:max_chars]


def extract_dialogue_text(payload: dict[str, Any]) -> str:
    turns = _turns_from_history_response(payload)
    if turns:
        return "\n".join(turns)
    turns = _turns_from_nested_messages(payload)
    if turns:
        return "\n".join(turns)
    for key in PREFERRED_TEXT_KEYS:
        if key in payload:
            val = payload[key]
            if isinstance(val, list):
                turns = _turns_from_list(val)
                if turns:
                    return "\n".join(turns)
            if isinstance(val, str):
                turns = _turns_from_string(val)
                if turns:
                    return "\n".join(turns)
            text = compact_text(val)
            if text:
                return text
    return compact_text(payload)


def dialogue_source(payload: dict[str, Any]) -> str | None:
    """Which recognized key supplied the dialogue, or None if we only have a fallback dump.

    `build_pilot_pairs` uses this to FAIL FAST: if no recognized key matched, the dataset
    schema is not understood and every pair would be poisoned — better to crash with guidance.
    """
    if _turns_from_history_response(payload):
        return "history+response"
    if _turns_from_nested_messages(payload):
        return "messages.message"
    for key in PREFERRED_TEXT_KEYS:
        if key not in payload:
            continue
        val = payload[key]
        if isinstance(val, list):
            if _turns_from_list(val):
                return key
        elif isinstance(val, str):
            if _turns_from_string(val) or compact_text(val):
                return key
        elif compact_text(val):
            return key
    return None


def truncate_to_turns(dialogue_text: str, turns_target: int) -> str:
    """Keep the first `turns_target` turns. Uses newline/role markers only.

    If no reliable turn markers are found, returns the text unchanged rather than
    doing fragile sentence surgery: splitting on ". " breaks abbreviations
    ("Dr. Smith", "U.S.") and corrupts the dialogue content itself.
    """
    if turns_target <= 0:
        return dialogue_text

    lines = [line.strip() for line in dialogue_text.splitlines() if line.strip()]
    if len(lines) >= turns_target:
        return "\n".join(lines[:turns_target])

    separators = (
        " User:", " Assistant:", " user:", " assistant:", " USER:", " ASSISTANT:",
        " 用户：", " 助手：", "用户:", "助手:",
    )
    normalized = dialogue_text
    for sep in separators:
        normalized = normalized.replace(sep, "\n" + sep.strip())
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if len(lines) >= turns_target:
        return "\n".join(lines[:turns_target])

    # No reliable turn markers -> return as-is (do not corrupt content).
    return dialogue_text


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(_stringify(item) for item in value)
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            parts.append(f"{key}: {_stringify(item)}")
        return "\n".join(parts)
    return str(value)
