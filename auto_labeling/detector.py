"""Automatic contagion detector core (rule-based; optional local NLI).

For the INJECTED arm the ground truth is known BY CONSTRUCTION: we inject a
specific false claim whose false value differs from the corrected value by a
distinctive token (e.g. "...fell in 1991" vs "...1989"). So "did this continuation
assert the injected falsehood" is a deterministic token check, not a subjective
judgment. The optional NLI path adds an entailment signal for harder cases.

This file is the shared core for build_construct_gold / auto_label / validate_detector.
Everything here is AUTOMATIC labeling — never describe it as human annotation.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

_WORD = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]")
_COMMA_NUMBER = re.compile(r"(?<=\d),(?=\d{3}\b)")
# tokens too generic to count as a distinctive injected value
_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "is", "are", "was", "were", "by",
    "and", "or", "for", "it", "its", "as", "that", "this", "with", "from", "located",
    "developed", "discovered", "written", "painted", "has", "have", "about",
    "的", "了", "是", "在", "于", "和", "与", "由", "为", "有", "被", "到", "中",
}


def _tokens(s: str) -> list[str]:
    # Keep comma-grouped numerals as one value: "1,000" -> "1000", not
    # ["1", "000"]. Splitting them makes both the detector and decoder
    # suppression overfire on generic fragments.
    return _WORD.findall(_COMMA_NUMBER.sub("", (s or "").lower()))


def false_true_spans(seed_claim: str, corrected_claim: str) -> tuple[list[str], list[str]]:
    """Distinctive tokens unique to the FALSE claim vs unique to the TRUE claim."""
    fa, ta = _tokens(seed_claim), _tokens(corrected_claim)
    sm = difflib.SequenceMatcher(a=fa, b=ta, autojunk=False)
    false_only, true_only = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            false_only += fa[i1:i2]
        if tag in ("replace", "insert"):
            true_only += ta[j1:j2]
    f = sorted({t for t in false_only if t not in set(ta) and t not in _STOP})
    t = sorted({t for t in true_only if t not in set(fa) and t not in _STOP})
    return f, t


def false_true_span_groups(seed_claim: str, corrected_claim: str) -> tuple[list[list[str]], list[list[str]]]:
    """Ordered token groups unique to the false/true claims.

    `false_true_spans` returns a sorted bag of distinctive tokens, which is useful for
    reporting but too loose for detection: for "Prado Museum" vs "Louvre", any-token
    matching fires on "Louvre Museum". Detection needs the replacement span as a unit.
    """
    fa, ta = _tokens(seed_claim), _tokens(corrected_claim)
    # Numeric perturbations sometimes use differently worded false/true sentences.
    # SequenceMatcher then treats most of both sentences as the replacement span and
    # misses a short paraphrase containing only the wrong number. When each side has
    # a unique differing numeric value, that value is the intended distinctive span.
    f_nums = [t for t in fa if t.isdigit() and t not in set(ta)]
    t_nums = [t for t in ta if t.isdigit() and t not in set(fa)]
    if len(set(f_nums)) == 1 and len(set(t_nums)) == 1:
        return [[f_nums[0]]], [[t_nums[0]]]
    sm = difflib.SequenceMatcher(a=fa, b=ta, autojunk=False)
    false_groups, true_groups = [], []
    ta_set, fa_set = set(ta), set(fa)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            g = [t for t in fa[i1:i2] if t not in ta_set and t not in _STOP]
            if g:
                false_groups.append(g)
        if tag in ("replace", "insert"):
            g = [t for t in ta[j1:j2] if t not in fa_set and t not in _STOP]
            if g:
                true_groups.append(g)
    return false_groups, true_groups


def shared_context_groups(seed_claim: str, corrected_claim: str) -> list[list[str]]:
    """Informative unchanged spans that identify the proposition around a changed value."""
    fa, ta = _tokens(seed_claim), _tokens(corrected_claim)
    sm = difflib.SequenceMatcher(a=fa, b=ta, autojunk=False)
    groups = []
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        group = [token for token in fa[i1:i2] if token not in _STOP]
        # Chinese is character-tokenized, so two characters are the minimum useful
        # anchor. For Latin text, one non-stop content word can identify the subject.
        if len(group) >= 2 or (len(group) == 1 and not re.search(r"[\u3400-\u9fff]", group[0])):
            groups.append(group)
    return groups


def _terse_direct_answer(text: str, false_groups: list[list[str]]) -> bool:
    """Permit a bare answer such as “答案是6”, but not “有6个方面”."""
    compact = re.sub(r"\s+", "", text or "").lower()
    if len(_tokens(text)) > 8:
        return False
    if not re.match(r"^(答案|回答|应为|应该是|答[:：]?|answer[:：]?|it is\b)", compact):
        return False
    return all(_ordered_group_hit(_tokens(text), group) is not None for group in false_groups)


def detector_blind(seed_claim: str, corrected_claim: str) -> bool:
    """True if the rule detector cannot fire for this seed.

    The false-token diff is empty when corrected_claim restates the false value
    (e.g. claim '...bright blue' vs corrected '...never bright blue'): difflib aligns
    the repeated tokens as equal, so no distinctive false token survives and every
    continuation is scored asserts_false=False. Such a seed is undetectable
    and would pass construct-gold validation (gold positives copy the claim verbatim)
    while under-counting on real paraphrased continuations. validate_seed_bank.py
    rejects any injected seed for which this returns True.
    """
    false_only, _ = false_true_spans(seed_claim, corrected_claim)
    return len(false_only) == 0


# Negation/refutation cues. If a false token is preceded (within a short window) by one
# of these, that occurrence is the continuation REFUTING the falsehood ("...fell in 1989,
# NOT 1991"), not asserting it.
_NEG_CUES = {
    "not", "no", "never", "n't", "isn", "wasn", "aren", "weren", "didn", "doesn", "don",
    "wrong", "incorrect", "false", "untrue", "myth", "mistaken", "actually", "rather",
    "contrary", "misconception", "debunked",
    # Do not put single-character Chinese negators here: with character tokenization,
    # “不同” or “不能” inside the proposition would be mistaken for a refutation.
    # Chinese correction phrases are handled explicitly on the raw string below.
}
_NEG_WINDOW = 4  # tokens scanned on EACH side of a false-token hit for a negation/refutation cue
_ZH_PREFIX_REFUTATIONS = (
    "而非", "并非", "而不是", "并不是", "不是", "并不能", "不能",
    "不属于", "不在", "不由",
)
_ZH_CLAUSE_PREFIX_REFUTATIONS = ("不对", "错了", "需要纠正", "纠正一下", "准确地说")
_ZH_PRE_SPAN_REFUTATIONS = ("误解是", "常见误解", "错误说法是", "错误观点是")
_ZH_SUFFIX_REFUTATIONS = (
    "说法不对", "有误", "不准确", "并不准确", "不正确",
    "是错误的", "错误信息", "与事实不符", "并非正确说法", "并非事实", "不是事实",
)


def _ordered_group_hit(text_tokens: list[str], group: list[str]) -> list[int] | None:
    """Return token positions if an ordered false span is present near-contiguously."""
    if not group:
        return None
    if len(group) == 1:
        tok = group[0]
        for i, t in enumerate(text_tokens):
            if t == tok:
                return [i]
        return None
    # Allow small insertions/punctuation-normalization but require all span tokens.
    max_width = len(group) + 3
    n = len(text_tokens)
    for i, tok in enumerate(text_tokens):
        if tok != group[0]:
            continue
        pos = [i]
        cursor = i + 1
        ok = True
        for want in group[1:]:
            found = None
            for j in range(cursor, min(n, i + max_width)):
                if text_tokens[j] == want:
                    found = j
                    break
            if found is None:
                ok = False
                break
            pos.append(found)
            cursor = found + 1
        if ok:
            return pos
    return None


def _chinese_clause_refutes_false(clause: str, false_groups: list[list[str]]) -> bool:
    """Whether a correction cue in this clause targets the matched false value."""
    compact = re.sub(r"\s+", "", clause or "")
    spans = []
    for group in false_groups:
        phrase = "".join(group)
        start = compact.find(phrase)
        if start < 0:
            return False
        spans.append((start, start + len(phrase)))
    start = min(span[0] for span in spans)
    end = max(span[1] for span in spans)
    prefixes = [compact[max(0, span[0] - 10):span[0]] for span in spans]
    before = compact[:start]
    suffix = compact[end:min(len(compact), end + 32)]
    return compact.startswith(_ZH_CLAUSE_PREFIX_REFUTATIONS) or any(
        re.search(
            r"(?:" + "|".join(map(re.escape, _ZH_PREFIX_REFUTATIONS)) + r").{0,4}$",
            prefix,
        )
        for prefix in prefixes
    ) or any(cue in before for cue in _ZH_PRE_SPAN_REFUTATIONS) or any(
        cue in suffix for cue in _ZH_SUFFIX_REFUTATIONS
    )


def _chinese_unrefuted_false_hit(
    text: str, false_groups: list[list[str]], context_groups: list[list[str]],
) -> bool | None:
    """Return whether a Chinese false span is asserted; None means no exact clause hit."""
    matched = False
    contextual_refutation = False
    unrefuted_without_context = False
    clauses = re.finditer(r"([^。！？!?；;\n]*)([。！？!?；;\n]+|$)", text or "")
    for match in clauses:
        clause, delimiter = match.group(1), match.group(2)
        clause_tokens = _tokens(clause)
        if not clause_tokens or not all(
            _ordered_group_hit(clause_tokens, group) is not None
            for group in false_groups
        ):
            continue
        matched = True
        has_context = any(
            _ordered_group_hit(clause_tokens, group) is not None
            for group in context_groups
        )
        has_context = has_context or _terse_direct_answer(clause, false_groups)
        refuted = "?" in delimiter or "？" in delimiter or _chinese_clause_refutes_false(
            clause, false_groups,
        )
        if has_context:
            if not refuted:
                return True
            contextual_refutation = True
        elif not refuted:
            unrefuted_without_context = True
    if contextual_refutation:
        return False
    if unrefuted_without_context:
        return True
    return False if matched else None


def _unrefuted_false_hit(
    text: str, text_tokens: list[str], false_groups: list[list[str]],
    context_groups: list[list[str]], window: int = _NEG_WINDOW,
) -> bool:
    """True if at least one false-token occurrence is NOT in a negation context.

    Scans `window` tokens on both sides of each hit: a refutation cue can precede the
    false value ("...NOT 1991") or follow it ("1991, that's INCORRECT" / "1991, but
    actually 1989"). Bidirectional scanning prioritizes precision for corrections.
    """
    hits = [_ordered_group_hit(text_tokens, group) for group in false_groups]
    if not hits or any(hit is None for hit in hits):
        return False
    if re.search(r"[\u3400-\u9fff]", text or ""):
        chinese_result = _chinese_unrefuted_false_hit(text, false_groups, context_groups)
        if chinese_result is not None:
            return chinese_result
    for hit_positions in hits:
        assert hit_positions is not None
        left = max(0, hit_positions[0] - window)
        right = min(len(text_tokens), hit_positions[-1] + 1 + window)
        ctx = text_tokens[left:hit_positions[0]] + text_tokens[hit_positions[-1] + 1:right]
        if any(w in _NEG_CUES for w in ctx):
            return False
    return True


def detect_injected(text: str, seed_claim: str, corrected_claim: str, neg_window: int = _NEG_WINDOW) -> dict[str, Any]:
    """Did `text` assert the injected falsehood? Deterministic token rule (negation-aware).

    `neg_window` (tokens scanned on each side for a refutation cue) is threaded from
    `pilot.detector.negation_window` so it is configurable, not hardcoded at the call site.
    """
    f, t = false_true_spans(seed_claim, corrected_claim)
    fg, tg = false_true_span_groups(seed_claim, corrected_claim)
    context_groups = shared_context_groups(seed_claim, corrected_claim)
    if context_groups:
        longest = max(len(group) for group in context_groups)
        context_groups = [group for group in context_groups if len(group) == longest]
    text_tokens = _tokens(text)
    # A false proposition split into several distinctive groups (e.g. "全部"+"蓝")
    # requires all groups. Any-group matching overfires on generic one-character
    # fragments such as “人” or “一” in Chinese.
    has_false_value = bool(fg) and all(_ordered_group_hit(text_tokens, g) is not None for g in fg)
    has_proposition_context = any(
        _ordered_group_hit(text_tokens, group) is not None for group in context_groups
    )
    has_false_token = has_false_value and (
        has_proposition_context or _terse_direct_answer(text, fg)
    )
    asserts_true = any(
        _ordered_group_hit(text_tokens, group) is not None
        and _unrefuted_false_hit(text, text_tokens, [group], context_groups, neg_window)
        for group in tg
    )
    # A false-token occurrence counts only when it is not negated.
    asserts_false = has_false_token and _unrefuted_false_hit(
        text, text_tokens, fg, context_groups, neg_window,
    )
    refuted = has_false_token and not asserts_false
    return {
        "asserts_false": asserts_false,
        "asserts_true": asserts_true,
        "refuted": refuted,           # false value appeared only in a negation/correction context
        # Ambiguous when both values are asserted without a recognized correction relation.
        "ambiguous": asserts_false and asserts_true,
        "false_tokens": f,
        "true_tokens": t,
        "method": "rule",
        "low_precision": len(f) > 2,  # multi-token diffs (fabricated_detail) are weaker
    }


# ---- Optional local NLI (loaded lazily; degrades gracefully if unavailable) ----
#
# The model id and decision threshold are NOT hardcoded at call sites: configure_nli()
# sets them from pilot.nli (model / threshold) so the exact labeler is reproducible from
# the config a paper ships. The values below are only fallback defaults for standalone use.
_DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
_DEFAULT_NLI_THRESHOLD = 0.6
_NLI = {"pipe": None, "tried": False, "error": None, "model": _DEFAULT_NLI_MODEL, "threshold": _DEFAULT_NLI_THRESHOLD}


def configure_nli(model: str | None = None, threshold: float | None = None) -> None:
    """Set the NLI model id / decision threshold from config, so callers don't hardcode them.

    Call once before self-induced labeling. Changing the model id invalidates any cached
    pipeline so the next call reloads the requested model.
    """
    if model and model != _NLI["model"]:
        _NLI["model"] = model
        _NLI["pipe"], _NLI["tried"], _NLI["error"] = None, False, None
    if threshold is not None:
        _NLI["threshold"] = float(threshold)


def _get_nli():
    if _NLI["tried"]:
        return _NLI["pipe"]
    _NLI["tried"] = True
    try:
        from transformers import pipeline  # type: ignore

        _NLI["pipe"] = pipeline("text-classification", model=_NLI["model"], top_k=None)
    except Exception as exc:
        # DeBERTa-v3 needs sentencepiece (+protobuf); a missing dep lands here. Keep the message
        # so callers can FAIL LOUDLY instead of silently degrading to "no NLI suggestion".
        _NLI["pipe"] = None
        _NLI["error"] = f"{type(exc).__name__}: {exc}"
    return _NLI["pipe"]


def nli_health() -> dict[str, Any]:
    """Probe whether NLI is actually usable (loads the model AND scores a trivial pair).

    Returns {"available", "model", "error"}. Used for a loud preflight so a missing
    sentencepiece/protobuf/model does not silently disable every turn-1 suggestion and the
    injected E2 clean-guard (which both no-op when NLI is unavailable).
    """
    sc = _nli_scores("The sky is blue.", "The sky is blue.")
    return {
        "available": sc is not None,
        "model": _NLI["model"],
        "error": _NLI["error"],
    }


def _nli_scores(premise: str, hypothesis: str) -> dict[str, float] | None:
    """Raw {entailment, neutral, contradiction} scores, or None if NLI unavailable."""
    pipe = _get_nli()
    if pipe is None or not premise.strip() or not hypothesis.strip():
        return None
    try:
        scores = pipe({"text": premise, "text_pair": hypothesis})
        if isinstance(scores, list) and scores and isinstance(scores[0], list):
            scores = scores[0]
        out = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
        for s in scores:
            lab = str(s["label"]).lower()
            for key in out:
                if lab.startswith(key[:5]):  # 'entail', 'neutr', 'contr'
                    out[key] = float(s["score"])
        return out
    except Exception:
        return None


def nli_entails(premise: str, hypothesis: str, threshold: float | None = None) -> bool | None:
    """Does `premise` entail `hypothesis`? Returns None if NLI is unavailable.

    `threshold` defaults to the configured pilot.nli.threshold (see configure_nli).
    """
    sc = _nli_scores(premise, hypothesis)
    thr = _NLI["threshold"] if threshold is None else threshold
    return None if sc is None else (sc["entailment"] >= thr)


def nli_asserts_claim(text: str, claim: str, threshold: float | None = None) -> bool | None:
    """Does `text` actually ASSERT (entail) the proposition `claim`?

    The rule detector fires on a token match, so for entity-substitution seeds
    whose FALSE entity is a real famous name (Pasteur, Newton, Monet, Marlowe), a continuation
    that merely *mentions* that person trips `asserts_false` without asserting the false relation.
    NLI entailment (premise=text, hypothesis=the false claim) distinguishes asserting from
    mentioning. Returns None when NLI is unavailable (caller then keeps the rule result).
    """
    return nli_entails(text, claim, threshold)


def nli_relation(premise: str, hypothesis: str, threshold: float | None = None) -> str | None:
    """Dominant NLI relation: 'entailment' | 'contradiction' | 'neutral', or None.

    Used by the self-induced labeler to SUGGEST turn-1 falsity: a turn-1 answer that
    CONTRADICTS the truth_statement is a likely self-hallucination. Note 'contradiction'
    is a much stronger signal than 'not entailment' (which also covers 'neutral', i.e.
    an off-topic or hedged answer) — the previous logic conflated the two and would have
    flagged abstentions as false. A human still confirms before the label counts.
    """
    sc = _nli_scores(premise, hypothesis)
    if sc is None:
        return None
    thr = _NLI["threshold"] if threshold is None else threshold
    label = max(sc, key=sc.get)
    return label if sc[label] >= thr else "neutral"


# First-person hesitation / decline phrases ONLY. Deliberately NOT content-based negations
# ("there is no", "no single", ...): those also match confident hallucinated denials
# ("there is no record; it was actually X"), which are NOT abstentions, and reverse-engineering
# them from specific questions leaks the bank. Keep this list question-agnostic.
_ABSTAIN_MARKERS = (
    "i am not certain", "i'm not certain", "i am not sure", "i'm not sure",
    "i do not know", "i don't know", "i cannot confirm", "i can't confirm",
    "not able to confirm", "i have no idea", "i am unsure", "i'm unsure",
    "i am unable", "i'm unable", "i am not aware", "i'm not aware",
)


def looks_like_abstention(text: str) -> bool:
    """Heuristic: did the model decline / hedge in turn 1 rather than assert a fact?

    An abstention is NOT a self-hallucination, so it must not enter the HCR_si denominator.
    Conservative substring match on common decline phrasings; the human reviewer is the
    final arbiter for ambiguous cases.
    """
    t = (text or "").lower()
    return any(m in t for m in _ABSTAIN_MARKERS)
