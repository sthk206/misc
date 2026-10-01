#!/usr/bin/env python3
"""
skill_eval_report.py - correctness + efficiency report for ONE skill version.

    python skill_eval_report.py baseline_runs/haiku --expected expected/ --out baseline.md

Expected layout (seed -> desk -> task):

    baseline_runs/haiku/run1/fxo/<task>/run_manifest.json   + artifacts
    baseline_runs/haiku/run1/ig-ca/<task>/...
    baseline_runs/haiku/run2/...

--expected points at a folder with the same <desk>/<task>/ layout holding the
reference artifacts. A single baseline run (e.g. baseline_runs/haiku/run1)
works too. Without it, only structural checks run (file exists, JSON parses,
text fields non-empty).

Outputs, next to --out:
    <out>.md                 the report
    <out>.json               the same numbers, machine-readable (for old-vs-new)
    <out>_judge_prompt.md    text fields to hand to an agent (only with --expected)

Standard library only.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path

# --------------------------------------------------------------------------
# Pricing. USD per million tokens (input, output); first substring match wins.
# CHECK THESE against current pricing for the models you run.
# --------------------------------------------------------------------------
PRICES = [
    ("haiku", (1.00, 5.00)),
    ("sonnet", (3.00, 15.00)),
    ("opus", (5.00, 25.00)),
]
CACHE_WRITE_5M = 1.25  # multiplier on the input price
CACHE_WRITE_1H = 2.00
CACHE_READ = 0.10

WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
DEFAULT_PRODUCTIVE = "Write,Edit,MultiEdit,NotebookEdit,Bash"
MIN_CACHED_PREFIX = 1024  # below this a "cache reset" is not meaningful

# User entries that are not real prompts (mirrors the Agent SDK's own filter).
NOT_A_PROMPT = re.compile(
    r"^(?:<local-command-stdout>|<session-start-hook>|<tick>|<goal>|<command-name>|"
    r"<system-reminder>|\[Request interrupted by user)"
)


# ==========================================================================
# 1. Trace parsing
# ==========================================================================
@dataclass
class Call:
    """One API response. Claude Code writes one JSONL line per content block,
    all sharing the same message.id and repeating `usage`, so lines are merged."""
    msg_id: str
    model: str
    chain: str          # which context this call belongs to (main or a subagent)
    main: bool
    seg: int            # index of the user query it belongs to
    ts: datetime | None
    tok: dict = field(default_factory=dict)
    tools: dict = field(default_factory=dict)  # tool_use_id -> (name, input)


@dataclass
class Seg:
    """Metrics for one user query within one run."""
    prompt: str = ""
    cost: float = 0.0
    tok: Counter = field(default_factory=Counter)
    api_calls: int = 0
    tool_calls: int = 0
    errors: int = 0
    dup_reads: int = 0
    first_action: int | None = None  # main-chain API calls before first productive tool
    wall: float | None = None
    resets: int = 0
    reset_tokens: int = 0
    tool_seq: list = field(default_factory=list)
    writes: set = field(default_factory=set)   # basenames written via Write/Edit
    bash: list = field(default_factory=list)   # Bash command strings

    @property
    def wasted(self) -> int:
        return self.errors + self.dup_reads


def parse_ts(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def read_jsonl(path: Path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                yield entry


def normalise_usage(u: dict) -> dict:
    """Flatten a usage object into the five token buckets that are priced."""
    split = u.get("cache_creation") or {}
    w5 = split.get("ephemeral_5m_input_tokens") or 0
    w1 = split.get("ephemeral_1h_input_tokens") or 0
    total_write = u.get("cache_creation_input_tokens") or 0
    if w5 + w1 < total_write:  # no split given (older traces): treat as 5-minute
        w5 += total_write - (w5 + w1)
    return {
        "input": u.get("input_tokens") or 0,
        "write_5m": w5,
        "write_1h": w1,
        "read": u.get("cache_read_input_tokens") or 0,
        "output": u.get("output_tokens") or 0,
    }


def prompt_text(entry: dict):
    """Return the text if this entry is a real user prompt, else None."""
    if entry.get("type") != "user":
        return None
    if entry.get("isMeta") or entry.get("isCompactSummary") or entry.get("isSidechain"):
        return None
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, str):
        texts = [content]
    elif isinstance(content, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return None
        texts = [b.get("text", "") for b in content
                 if isinstance(b, dict) and b.get("type") == "text"]
    else:
        return None
    for t in texts:
        t = t.strip()
        if t and not NOT_A_PROMPT.match(t):
            return t
    return None


def price_for(model: str, unpriced: set):
    for key, price in PRICES:
        if key in (model or "").lower():
            return price
    unpriced.add(model or "(none)")
    return dict(PRICES)["sonnet"]


def call_cost(tok: dict, model: str, unpriced: set) -> float:
    p_in, p_out = price_for(model, unpriced)
    weighted_in = (tok["input"] + tok["write_5m"] * CACHE_WRITE_5M
                   + tok["write_1h"] * CACHE_WRITE_1H + tok["read"] * CACHE_READ)
    return (weighted_in * p_in + tok["output"] * p_out) / 1e6


def load_segments(transcripts: list, productive: set, unpriced: set) -> list:
    """Parse all transcripts of one task run into per-query segments."""
    calls: dict[str, Call] = {}       # keyed by message.id: dedupes across lines AND files
    errors: dict[str, bool] = {}      # tool_use_id -> is_error
    prompts: list = []                # (timestamp, text) of each real user prompt
    seg_span: dict = defaultdict(lambda: [None, None])

    def absorb(entry, chain, main, seg):
        msg = entry.get("message") or {}
        if entry.get("type") == "user" and isinstance(msg.get("content"), list):
            for b in msg["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    errors[b.get("tool_use_id")] = bool(b.get("is_error"))
        if entry.get("type") != "assistant" or not isinstance(msg.get("usage"), dict):
            return
        if msg.get("model") == "<synthetic>":   # locally generated, no API call
            return
        mid = msg.get("id") or entry.get("uuid")
        call = calls.get(mid)
        if call is None:
            call = calls[mid] = Call(mid, msg.get("model", ""), chain, main, seg,
                                     parse_ts(entry.get("timestamp")),
                                     dict.fromkeys(("input", "write_5m", "write_1h",
                                                    "read", "output"), 0))
        # usage repeats on every line; earlier lines can carry a partial
        # output_tokens, so keep the field-wise maximum.
        for k, v in normalise_usage(msg["usage"]).items():
            call.tok[k] = max(call.tok[k], v)
        for b in msg.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                call.tools[b.get("id")] = (b.get("name", "?"), b.get("input") or {})

    # Pass 1: main transcripts, in order. Query boundaries come from real prompts.
    for path in transcripts:
        for entry in read_jsonl(path):
            ts = parse_ts(entry.get("timestamp"))
            text = prompt_text(entry)
            if text is not None:
                prompts.append((ts, text))
            if entry.get("type") in ("user", "assistant"):
                if not prompts:
                    prompts.append((ts, "(no user prompt found)"))
                seg = len(prompts) - 1
                if ts:
                    span = seg_span[seg]
                    span[0] = span[0] or ts
                    span[1] = ts
                side = bool(entry.get("isSidechain"))
                chain = f"{path}:{entry.get('agentId', 'side') if side else 'main'}"
                absorb(entry, chain, not side, seg)

    # Pass 2: subagent transcripts live in <session-id>/subagents/**/agent-*.jsonl.
    # They are assigned to a query by timestamp.
    starts = [p[0] for p in prompts]
    for path in transcripts:
        sub_dir = path.with_suffix("") / "subagents"
        for sub in sorted(sub_dir.rglob("*.jsonl")) if sub_dir.is_dir() else []:
            for entry in read_jsonl(sub):
                ts = parse_ts(entry.get("timestamp"))
                seg = len(prompts) - 1
                if ts and all(starts):
                    seg = max(0, bisect.bisect_right(starts, ts) - 1)
                if prompts:
                    absorb(entry, str(sub), False, seg)

    segs = [Seg(prompt=text) for _, text in prompts]
    last_in_chain: dict[str, Call] = {}
    main_calls_in_seg: Counter = Counter()
    seen_reads: dict[str, str] = {}   # read signature -> file path

    for call in calls.values():       # dicts keep first-seen order
        seg = segs[call.seg]
        seg.api_calls += 1
        seg.tok.update(call.tok)
        seg.cost += call_cost(call.tok, call.model, unpriced)

        # Cache reset: the previous call left a cached prefix, but this call
        # read less than half of it back (TTL expiry, compaction, prefix change).
        prev = last_in_chain.get(call.chain)
        if prev is not None:
            cached = prev.tok["read"] + prev.tok["write_5m"] + prev.tok["write_1h"]
            if cached >= MIN_CACHED_PREFIX and call.tok["read"] < 0.5 * cached:
                seg.resets += 1
                seg.reset_tokens += min(cached, call.tok["write_5m"] + call.tok["write_1h"])
        last_in_chain[call.chain] = call

        for tool_id, (name, inp) in call.tools.items():
            seg.tool_calls += 1
            seg.tool_seq.append(name)
            seg.errors += bool(errors.get(tool_id))
            path = str(inp.get("file_path") or inp.get("notebook_path") or "")
            if name == "Read":
                sig = json.dumps(inp, sort_keys=True)
                seg.dup_reads += sig in seen_reads
                seen_reads[sig] = path
            elif name in WRITE_TOOLS and path:
                seg.writes.add(Path(path).name)
                for sig in [s for s, p in seen_reads.items() if p == path]:
                    del seen_reads[sig]   # re-reading after an edit is legitimate
            elif name == "Bash":
                seg.bash.append(str(inp.get("command", "")))
            if call.main and seg.first_action is None and name in productive:
                seg.first_action = main_calls_in_seg[call.seg]
        if call.main:
            main_calls_in_seg[call.seg] += 1

    for i, seg in enumerate(segs):
        start, end = seg_span[i]
        if start and end:
            seg.wall = (end - start).total_seconds()
    return segs


def find_transcripts(manifest_path: Path):
    """Collect every *.jsonl path mentioned in the manifest (any key, any depth).
    Keys containing 'transcript' win if present."""
    task_dir = manifest_path.parent
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        manifest = {}
    found = []

    def walk(value, key=""):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{key}.{k}")
        elif isinstance(value, list):
            for v in value:
                walk(v, key)
        elif isinstance(value, str) and value.strip().endswith(".jsonl"):
            found.append((key.lower(), value.strip()))

    walk(manifest)
    wanted = [v for k, v in found if "transcript" in k] or [v for _, v in found]
    paths, missing = [], []
    for raw in dict.fromkeys(wanted):
        options = [Path(raw).expanduser(), task_dir / raw, task_dir / Path(raw).name]
        hit = next((p for p in options if p.is_file()), None)
        (paths if hit else missing).append(hit or raw)
    if not paths and not missing:
        paths = sorted(task_dir.glob("*.jsonl"))
    return paths, missing


# ==========================================================================
# 2. Artifact checks
# ==========================================================================
@dataclass
class Check:
    passed: bool = True
    fields: tuple = (0, 0)     # (matched, expected)
    numbers: tuple = (0, 0)
    texts: tuple = (0, 0)      # (non-empty, expected)
    issues: list = field(default_factory=list)
    judge: list = field(default_factory=list)   # (field, expected_text, actual_text)


def flatten(obj, prefix=""):
    """{'a': {'b': [1, 2]}} -> ('a.b[0]', 1), ('a.b[1]', 2)"""
    if isinstance(obj, dict) and obj:
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list) and obj:
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix or "(root)", obj


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_json(actual, expected, rtol, atol) -> Check:
    c = Check()
    act = dict(flatten(actual))
    if expected is None:                       # structural mode
        texts = {k: v for k, v in act.items() if isinstance(v, str)}
        empty = [k for k, v in texts.items() if not v.strip()]
        c.fields = (len(act), len(act))
        c.texts = (len(texts) - len(empty), len(texts))
        c.issues = [f"empty text: {k}" for k in empty]
        c.passed = not c.issues
        return c

    exp = dict(flatten(expected))
    f_ok = n_ok = n_all = t_ok = t_all = 0
    for key, want in exp.items():
        if key not in act:
            c.issues.append(f"missing field: {key}")
            n_all += is_number(want)
            t_all += isinstance(want, str)
            continue
        f_ok += 1
        got = act[key]
        if is_number(want):
            n_all += 1
            if is_number(got) and math.isclose(got, want, rel_tol=rtol, abs_tol=atol):
                n_ok += 1
            else:
                c.issues.append(f"number mismatch: {key} expected {want!r}, got {got!r}")
        elif isinstance(want, str):
            t_all += 1
            if isinstance(got, str) and (got.strip() or not want.strip()):
                t_ok += 1
                if got != want:
                    c.judge.append((key, want, got))
            else:
                c.issues.append(f"empty or non-text: {key} (got {got!r})")
        elif got != want:
            c.issues.append(f"value mismatch: {key} expected {want!r}, got {got!r}")
    c.issues += [f"extra field: {k}" for k in act if k not in exp]
    c.fields, c.numbers, c.texts = (f_ok, len(exp)), (n_ok, n_all), (t_ok, t_all)
    c.passed = not c.issues
    return c


NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def numbers_in(text):
    out = Counter()
    for tok in NUM_RE.findall(text):
        try:
            out[float(tok.replace(",", ""))] += 1
        except ValueError:
            pass
    return out


def check_text_file(actual: str, expected) -> Check:
    """Markdown / other text: gate is non-empty. Number overlap is informational."""
    c = Check(texts=(int(bool(actual.strip())), 1))
    if not actual.strip():
        c.issues.append("file is empty")
    if expected is not None:
        want, got = numbers_in(expected), numbers_in(actual)
        c.numbers = (sum((want & got).values()), sum(want.values()))
        if actual.strip() and actual != expected:
            c.judge.append(("(whole document)", expected, actual))
    c.passed = not c.issues
    return c


def check_artifact(actual: Path, expected, rtol, atol) -> Check:
    if not actual.is_file():
        return Check(passed=False, issues=["artifact missing"])
    text = actual.read_text(encoding="utf-8", errors="replace")
    exp_text = expected.read_text(encoding="utf-8", errors="replace") if expected else None
    if actual.suffix.lower() == ".json":
        try:
            return check_json(json.loads(text), json.loads(exp_text) if exp_text else None,
                              rtol, atol)
        except json.JSONDecodeError as e:
            return Check(passed=False, issues=[f"invalid JSON: {e}"])
    return check_text_file(text, exp_text)


def list_artifacts(task_dir: Path) -> set:
    if not task_dir.is_dir():
        return set()
    return {
        str(p.relative_to(task_dir)) for p in task_dir.rglob("*")
        if p.is_file() and p.name != "run_manifest.json" and p.suffix != ".jsonl"
        and not any(part.startswith(".") for part in p.relative_to(task_dir).parts)
    }


# ==========================================================================
# 3. Aggregation helpers
# ==========================================================================
def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def top(xs):
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def spread(xs):
    """max / min across seeds: the seed-to-seed noise for this quantity."""
    xs = [x for x in xs if x]
    return max(xs) / min(xs) if len(xs) > 1 else None


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def consistency(seqs):
    """1.0 = every seed made the same tool calls in the same order."""
    pairs = list(combinations(seqs, 2))
    if not pairs:
        return None
    sims = [1 - edit_distance(a, b) / max(len(a), len(b), 1) for a, b in pairs]
    return sum(sims) / len(sims)


def usd(x):
    return "–" if x is None else (f"${x:.4f}" if x < 1 else f"${x:.2f}")


def num(x, digits=1):
    if x is None:
        return "–"
    return str(int(x)) if float(x).is_integer() else f"{x:.{digits}f}"


def ratio(pair):
    return f"{pair[0]}/{pair[1]}" if pair[1] else "–"


def pct(pairs):
    done, total = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    return f"{100 * done / total:.0f}%" if total else "–"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ==========================================================================
# 4. Main
# ==========================================================================
@dataclass
class TaskRun:
    segs: list
    checks: dict          # artifact name -> Check
    notes: list

    @property
    def passed(self):
        return bool(self.checks) and all(c.passed for c in self.checks.values())

    def total(self, attr):
        """Sum over queries; None when there is no transcript (so it is left
        out of medians instead of counting as zero)."""
        if not self.segs:
            return None
        return sum(getattr(s, attr) or 0 for s in self.segs)


def artifact_query(run: TaskRun, name: str):
    """Best effort: the last query that wrote this file (Write/Edit path, or
    the filename appearing in a Bash command)."""
    base, hit = Path(name).name, None
    for i, seg in enumerate(run.segs):
        if base in seg.writes or any(base in cmd for cmd in seg.bash):
            hit = i
    return hit


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="folder containing one sub-folder per seed")
    ap.add_argument("--expected", type=Path, help="reference artifacts, <desk>/<task>/ layout")
    ap.add_argument("--out", type=Path, help="report path (default: <root-name>_report.md)")
    ap.add_argument("--rtol", type=float, default=1e-6, help="relative tolerance for numbers")
    ap.add_argument("--atol", type=float, default=1e-9, help="absolute tolerance for numbers")
    ap.add_argument("--max-text-chars", type=int, default=2000,
                    help="truncate each text field in the judge prompt")
    ap.add_argument("--productive-tools", default=DEFAULT_PRODUCTIVE,
                    help="tools that count as the 'first productive action'")
    args = ap.parse_args()
    out = args.out or Path(f"{args.root.resolve().name}_report.md")
    productive = {t.strip() for t in args.productive_tools.split(",") if t.strip()}
    unpriced: set = set()

    # ---- load every (seed, desk, task) -----------------------------------
    seeds = sorted(p.name for p in args.root.iterdir() if p.is_dir())
    task_dirs = defaultdict(dict)    # (desk, task) -> {seed: dir}
    for seed in seeds:
        for manifest in sorted((args.root / seed).rglob("run_manifest.json")):
            rel = manifest.parent.relative_to(args.root / seed).parts
            desk, task = rel[0], "/".join(rel[1:]) or rel[0]
            task_dirs[(desk, task)][seed] = manifest.parent
    if not task_dirs:
        raise SystemExit(f"No run_manifest.json found under {args.root}")

    runs = defaultdict(dict)         # (desk, task) -> {seed: TaskRun}
    for key, by_seed in task_dirs.items():
        exp_dir = args.expected / key[0] / key[1] if args.expected else None
        if exp_dir is not None and not exp_dir.is_dir():
            exp_dir = None
        names = list_artifacts(exp_dir) if exp_dir else \
            set().union(*(list_artifacts(d) for d in by_seed.values()))
        for seed, task_dir in by_seed.items():
            transcripts, missing = find_transcripts(task_dir / "run_manifest.json")
            notes = [f"transcript not found: {m}" for m in missing]
            if args.expected and exp_dir is None:
                notes.append("no expected folder for this task; structural checks only")
            if not transcripts:
                notes.append("no transcript; efficiency metrics unavailable")
            checks = {n: check_artifact(task_dir / n, exp_dir / n if exp_dir else None,
                                        args.rtol, args.atol) for n in sorted(names)}
            runs[key][seed] = TaskRun(load_segments(transcripts, productive, unpriced),
                                      checks, notes)

    # ---- per-task sections -------------------------------------------------
    body, judge_items, metrics = [], [], {"seeds": seeds, "tasks": {}}
    desk_rows = defaultdict(list)    # desk -> list of per-task summaries
    query_spreads = []

    for (desk, task), by_seed in sorted(runs.items()):
        present = [s for s in seeds if s in by_seed]
        n_pass = sum(by_seed[s].passed for s in present)
        all_pass = n_pass == len(seeds)
        lines = [f"### {desk} / {task} — {'PASS' if all_pass else 'FAIL'} "
                 f"({n_pass}/{len(seeds)} seeds)", ""]

        # query table
        n_q = max((len(by_seed[s].segs) for s in present), default=0)
        counts = {len(by_seed[s].segs) for s in present}
        art_q = {name: next((q for s in present
                             if (q := artifact_query(by_seed[s], name)) is not None), None)
                 for name in by_seed[present[0]].checks}
        rows, q_metrics = [], []
        for q in range(n_q):
            segs = [by_seed[s].segs[q] for s in present if q < len(by_seed[s].segs)]
            costs = [s.cost for s in segs]
            sp = spread(costs)
            if sp:
                query_spreads.append(sp)
            mine = [n for n, qq in art_q.items() if qq == q]
            ok = sum(all(by_seed[s].checks[n].passed for s in present) for n in mine)
            cons = consistency([s.tool_seq for s in segs])
            prompt = re.sub(r"\s+", " ", segs[0].prompt)[:48].replace("|", "/")
            rows.append([f"Q{q + 1}", prompt, f"{ok}/{len(mine)}" if mine else "–",
                         usd(med(costs)), usd(top(costs)), f"{sp:.1f}x" if sp else "–",
                         num(med(s.api_calls for s in segs)),
                         num(med(s.tool_calls for s in segs)),
                         num(med(s.wasted for s in segs)),
                         num(med(s.first_action for s in segs)),
                         num(med(s.wall for s in segs), 0),
                         f"{cons:.2f}" if cons is not None else "–",
                         sum(s.resets for s in segs)])
            q_metrics.append({"cost_median": med(costs), "cost_max": top(costs),
                              "cost_spread": sp, "api_calls": med(s.api_calls for s in segs),
                              "tool_calls": med(s.tool_calls for s in segs),
                              "wasted": med(s.wasted for s in segs),
                              "consistency": cons})
        task_costs = [by_seed[s].total("cost") for s in present]
        if n_q:
            rows.append(["**Task**", "", f"{n_pass}/{len(seeds)} seeds",
                         usd(med(task_costs)), usd(top(task_costs)),
                         f"{spread(task_costs):.1f}x" if spread(task_costs) else "–",
                         num(med(by_seed[s].total("api_calls") for s in present)),
                         num(med(by_seed[s].total("tool_calls") for s in present)),
                         num(med(by_seed[s].total("wasted") for s in present)), "",
                         num(med(by_seed[s].total("wall") for s in present), 0), "",
                         sum(by_seed[s].total("resets") or 0 for s in present)])
            lines += [table(["Query", "Prompt", "Artifacts ok", "Cost med", "Cost max",
                             "Seed spread", "API calls", "Tool calls", "Wasted",
                             "Calls before 1st action", "Wall s", "Consistency",
                             "Cache resets"], rows), ""]
        if len(counts) > 1:
            lines += [f"> Seeds have different query counts {sorted(counts)}; "
                      "queries are aligned by position.", ""]

        # artifact table
        art_rows = []
        for name in by_seed[present[0]].checks:
            cs = {s: by_seed[s].checks[name] for s in present}
            q = art_q[name]
            art_rows.append([f"`{name}`", f"Q{q + 1}" if q is not None else "–"]
                            + ["no run" if s not in cs else "ok" if cs[s].passed
                               else f"FAIL ({len(cs[s].issues)})" for s in seeds]
                            + [pct([c.fields for c in cs.values()]),
                               pct([c.numbers for c in cs.values()]),
                               pct([c.texts for c in cs.values()])])
            for s, c in cs.items():
                for fld, want, got in c.judge:
                    judge_items.append((f"{desk}/{task}/{name} :: {fld}", s, want, got))
        if art_rows:
            lines += [table(["Artifact", "Query"] + seeds
                            + ["Fields", "Numbers", "Text non-empty"], art_rows), ""]
        for s in present:
            for name, c in by_seed[s].checks.items():
                if c.issues:
                    more = f" (+{len(c.issues) - 5} more)" if len(c.issues) > 5 else ""
                    lines.append(f"- **{s}** `{name}`: " + "; ".join(c.issues[:5]) + more)
            lines += [f"- **{s}** note: {n}" for n in by_seed[s].notes]
        body.append("\n".join(lines).rstrip() + "\n")

        desk_rows[desk].append({"all_pass": all_pass, "n_pass": n_pass, "runs": by_seed})
        metrics["tasks"][f"{desk}/{task}"] = {
            "pass_all_seeds": all_pass, "seeds_passed": n_pass,
            "cost_median": med(task_costs), "cost_max": top(task_costs),
            "cost_by_seed": {s: by_seed[s].total("cost") for s in present},
            "queries": q_metrics,
        }

    # ---- summary -------------------------------------------------------------
    def summarise(label, tasks):
        task_runs = [r for t in tasks for r in t["runs"].values()]
        passed = sum(t["n_pass"] for t in tasks)
        total_cost = sum(r.total("cost") or 0 for r in task_runs)
        per_seed = lambda attr: [sum(t["runs"][s].total(attr) or 0
                                     for t in tasks if s in t["runs"]) for s in seeds]
        row = {"pass_k": f"{sum(t['all_pass'] for t in tasks)}/{len(tasks)}",
               "task_runs_passed": f"{passed}/{len(tasks) * len(seeds)}",
               "cost_per_success": total_cost / passed if passed else None,
               "cost_per_seed_median": med(per_seed("cost")),
               "cost_per_seed_max": top(per_seed("cost")),
               "api_calls": med(per_seed("api_calls")),
               "tool_calls": med(per_seed("tool_calls")),
               "wasted": med(per_seed("wasted")),
               "cache_resets": sum(r.total("resets") or 0 for r in task_runs),
               "reset_tokens": sum(r.total("reset_tokens") or 0 for r in task_runs)}
        metrics.setdefault("summary", {})[label] = row
        return [label, row["pass_k"], row["task_runs_passed"], usd(row["cost_per_success"]),
                usd(row["cost_per_seed_median"]), usd(row["cost_per_seed_max"]),
                num(row["api_calls"]), num(row["tool_calls"]), num(row["wasted"]),
                f"{row['cache_resets']} ({row['reset_tokens'] / 1000:.1f}k tok)"]

    all_tasks = [t for d in desk_rows.values() for t in d]
    summary = [summarise(d, desk_rows[d]) for d in sorted(desk_rows)]
    summary.append(summarise("**ALL**", all_tasks))
    noise = med(query_spreads)
    no_trace = sum(not r.segs for t in all_tasks for r in t["runs"].values())
    metrics["noise_median_spread"] = noise

    head = [
        f"# Skill eval report: `{args.root}`", "",
        f"- Seeds: {', '.join(seeds)}",
        f"- Expected artifacts: `{args.expected}`" if args.expected
        else "- Expected artifacts: none given (structural checks only)",
        f"- Number tolerance: rtol={args.rtol}, atol={args.atol}",
        f"- Generated: {datetime.now():%Y-%m-%d %H:%M}", "",
        "## Summary", "",
        table(["Desk", "Tasks passing all seeds", "Task-runs passed",
               "Cost per successful run", "Cost/seed med", "Cost/seed max", "API calls",
               "Tool calls", "Wasted calls", "Cache resets"], summary), "",
        "**Decision metrics:** *Tasks passing all seeds* (the gate), then "
        "*Cost per successful run* (total cost of every task-run, divided by passing task-runs).",
        "",
        f"**Noise floor:** median seed spread (max/min cost of the same query) is "
        f"{f'{noise:.2f}x' if noise else 'n/a'}"
        + (f", worst {max(query_spreads):.1f}x" if query_spreads else "")
        + ". A difference between two skills smaller than this is not evidence.", "",
    ]
    if no_trace:
        head += [f"**Warning:** {no_trace} task-run(s) have no transcript, so the cost "
                 "figures above undercount. See the notes under each task.", ""]
    legend = [
        "## How to read this", "",
        "- **Cost**: USD from the price table at the top of the script. Per API call: "
        "input + 1.25x 5-min cache writes + 2x 1-hour cache writes + 0.1x cache reads, "
        "plus output at the output price. Lines sharing a message id are merged; "
        "subagent transcripts are included.",
        "- **Wasted**: tool calls that returned an error, plus repeat `Read`s of a file "
        "that had not been edited since the last read.",
        "- **Calls before 1st action**: main-agent API calls before the first "
        f"{'/'.join(sorted(productive))}.",
        "- **Consistency**: 1.0 means all seeds made the same tool calls in the same order "
        "(1 minus normalised edit distance, averaged over seed pairs).",
        "- **Cache resets**: calls that read back under half of the prefix the previous "
        "call had cached (expiry, compaction, or a changed prefix). They inflate cost "
        "for reasons unrelated to the skill.",
        "- **Artifacts**: JSON must have the same fields, matching numbers and non-empty "
        "text. For `.md`, the gate is non-empty; the Numbers column is informational. "
        "**Query** is a best-effort guess from Write/Edit/Bash calls.",
    ]
    if unpriced:
        legend.append(f"- **Warning:** no price for {sorted(unpriced)}; Sonnet prices used.")

    report = "\n".join(head) + "\n## Tasks\n\n" + "\n".join(body) + "\n" + "\n".join(legend) + "\n"
    out.write_text(report, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"report : {out}\nmetrics: {out.with_suffix('.json')}")

    # ---- judge prompt ----------------------------------------------------------
    if judge_items:
        cut = lambda t: t if len(t) <= args.max_text_chars \
            else t[:args.max_text_chars] + " …[truncated]"
        parts = [
            "# Text-field comparison", "",
            "Each item below is an LLM-generated text field: a reference version and the "
            "version one run produced. Wording will differ; judge the content.", "",
            "- EQUIVALENT: same facts and conclusions.",
            "- MINOR: small omissions or shifts of emphasis, nothing contradicts the reference.",
            "- DIFFERENT: missing key facts, or numbers or conclusions that conflict.", "",
            "Numbers quoted inside the text must agree with the reference. Reply with a JSON "
            'array only: `[{"item": 1, "verdict": "...", "reason": "one sentence"}]`.', "",
        ]
        for i, (where, seed, want, got) in enumerate(judge_items, 1):
            parts += [f"## Item {i} — {where} — {seed}",
                      f"<expected>\n{cut(want)}\n</expected>",
                      f"<actual>\n{cut(got)}\n</actual>", ""]
        judge_path = out.with_name(out.stem + "_judge_prompt.md")
        judge_path.write_text("\n".join(parts), encoding="utf-8")
        print(f"judge  : {judge_path} ({len(judge_items)} items)")


if __name__ == "__main__":
    main()
