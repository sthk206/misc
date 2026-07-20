"""Hard-negative generators over composed facts.

Each generator returns a corrupted variant differing from the input in exactly
one intended way (or None when not applicable):
  - date/number swap  (regex-detected, perturbed)
  - negation insertion/removal (rule-based; LLM-assisted fallback lives in
    `negate_llm`, verified by a template check)
  - entity swap       (spaCy NER when available, else capitalized-span heuristic)
  - role swap         ("A hired B" -> "B hired A")
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from lcm_mem.llm import prompts
from lcm_mem.llm.gateway import CachedGateway

_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_NUMBER = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")
_MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]
_MONTH_RE = re.compile(r"\b(" + "|".join(_MONTHS) + r")\b")

# copulas/auxiliaries where "not" can be inserted directly after
_NEGATABLE = re.compile(
    r"\b(is|are|was|were|has|have|had|can|could|will|would|does|did|do)\b(?! not\b)(?!n't)"
)
_NOT = re.compile(r"\b(is|are|was|were|has|have|had|can|could|will|would|does|did|do) not\b")

_ROLE_VERBS = re.compile(
    r"^(?P<a>.+?)\s+(?P<v>hired|married|founded|acquired|defeated|succeeded|"
    r"preceded|replaced|killed|taught|influenced|employed|directed)\s+(?P<b>.+?)(?P<tail>[.]?)$"
)

_CAP_SPAN = re.compile(r"\b(?!The\b|A\b|An\b|In\b|On\b|Of\b)[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")


@dataclass
class HardNegative:
    text: str
    kind: str  # date_swap | number_swap | negation | entity_swap | role_swap


def swap_date(text: str, rng: random.Random) -> str | None:
    m = _YEAR.search(text)
    if m:
        year = int(m.group(0))
        delta = rng.choice([-7, -5, -3, -2, 2, 3, 5, 7])
        return text[: m.start()] + str(year + delta) + text[m.end():]
    m = _MONTH_RE.search(text)
    if m:
        others = [mo for mo in _MONTHS if mo != m.group(0)]
        return text[: m.start()] + rng.choice(others) + text[m.end():]
    return None


def swap_number(text: str, rng: random.Random) -> str | None:
    # skip years — those belong to swap_date
    for m in _NUMBER.finditer(text):
        if _YEAR.fullmatch(m.group(0)):
            continue
        raw = m.group(0).replace(",", "")
        if "." in raw:
            val = float(raw)
            new = val * rng.choice([0.5, 1.5, 2.0]) + rng.choice([1, 2])
            new_s = f"{new:g}"
        else:
            val = int(raw)
            delta = max(1, abs(val) // 3)
            new_s = str(val + rng.choice([-delta, delta]) or val + 1)
        if new_s == m.group(0):
            continue
        return text[: m.start()] + new_s + text[m.end():]
    return None


def negate(text: str) -> str | None:
    """Rule-based negation flip: remove an existing 'not', else insert one
    after the first negatable auxiliary/copula."""
    m = _NOT.search(text)
    if m:
        return text[: m.start()] + m.group(1) + text[m.end():]
    m = _NEGATABLE.search(text)
    if m:
        return text[: m.end()] + " not" + text[m.end():]
    return None


def negate_llm(text: str, gateway: CachedGateway, model: str) -> str | None:
    """LLM-assisted negation for sentences the rules cannot handle, verified by
    a template check (the output must differ and contain/lose a negator)."""
    out = gateway.chat(
        [{"role": "user", "content": prompts.NEGATION_REWRITE_V1.format(sentence=text)}],
        model=model,
    ).strip()
    if out == text:
        return None
    had_neg = bool(re.search(r"\bnot\b|n't\b|\bnever\b|\bno\b", text))
    has_neg = bool(re.search(r"\bnot\b|n't\b|\bnever\b|\bno\b", out))
    return out if had_neg != has_neg else None


def find_entities(text: str) -> list[tuple[str, str]]:
    """Return (surface, label) entity mentions. Uses spaCy if installed, else a
    capitalized-span heuristic with label 'SPAN'."""
    try:
        import spacy

        if not hasattr(find_entities, "_nlp"):
            find_entities._nlp = spacy.load("en_core_web_sm")  # type: ignore[attr-defined]
        doc = find_entities._nlp(text)  # type: ignore[attr-defined]
        return [(e.text, e.label_) for e in doc.ents]
    except (ImportError, OSError):
        return [(m.group(0), "SPAN") for m in _CAP_SPAN.finditer(text)]


def swap_entity(
    text: str,
    entity_pool: dict[str, list[str]],
    rng: random.Random,
) -> str | None:
    """Replace one entity mention with a different same-type entity from the
    corpus pool ({label: [names]})."""
    ents = find_entities(text)
    rng.shuffle(ents)
    for surface, label in ents:
        candidates = [e for e in entity_pool.get(label, []) if e != surface and e not in text]
        if candidates:
            return text.replace(surface, rng.choice(candidates), 1)
    return None


def role_swap(text: str) -> str | None:
    m = _ROLE_VERBS.match(text.strip())
    if not m:
        return None
    return f"{m.group('b')} {m.group('v')} {m.group('a')}{m.group('tail')}"


def build_entity_pool(texts: list[str]) -> dict[str, list[str]]:
    pool: dict[str, set[str]] = {}
    for t in texts:
        for surface, label in find_entities(t):
            pool.setdefault(label, set()).add(surface)
    return {k: sorted(v) for k, v in pool.items()}


def generate_hard_negatives(
    text: str,
    entity_pool: dict[str, list[str]] | None = None,
    rng: random.Random | None = None,
) -> list[HardNegative]:
    rng = rng or random.Random(0)
    out: list[HardNegative] = []
    for fn, kind in (
        (lambda t: swap_date(t, rng), "date_swap"),
        (lambda t: swap_number(t, rng), "number_swap"),
        (negate, "negation"),
        (role_swap, "role_swap"),
    ):
        neg = fn(text)
        if neg is not None and neg != text:
            out.append(HardNegative(neg, kind))
    if entity_pool:
        neg = swap_entity(text, entity_pool, rng)
        if neg is not None and neg != text:
            out.append(HardNegative(neg, "entity_swap"))
    return out


def fluency_filter(
    negatives: list[str], gateway: CachedGateway, model: str, batch: int = 25
) -> list[bool]:
    """One cached LLM pass marking each candidate fluent (True) or not."""
    keep: list[bool] = []
    for i in range(0, len(negatives), batch):
        chunk = negatives[i : i + batch]
        numbered = "\n".join(f"{j + 1}. {s}" for j, s in enumerate(chunk))
        verdicts = gateway.chat_json(
            [{"role": "user", "content": prompts.FLUENCY_CHECK_V1.format(
                numbered_sentences=numbered)}],
            model=model,
        )
        keep.extend([str(v).upper().startswith("Y") for v in verdicts])
    return keep
