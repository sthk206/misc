"""Deterministic mock-atom fixtures: reference data, snapshots, DoD moves, and
a trade-lifecycle feed. Everything is seeded — same seed, same world.

Atom is the source of truth for numbers; nothing here is ever memorized by the
stores except as immutable snapshot references (snapshot_id + hash).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field, asdict

DESKS = {
    "FX Options Desk": "Global Markets",
    "G10 Spot Desk": "Global Markets",
    "EM Desk": "Global Markets",
    "Rates Desk": "Treasury",
    "Exotics Desk": "Global Markets",
}
PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "EURJPY", "EURGBP"]
CLIENTS = ["CL-1001", "CL-1002", "CL-1003", "CL-2001", "CL-2002"]


@dataclass
class TradeEvent:
    event_id: str
    day: int                      # simulation day index
    kind: str                     # open | amend | cancel | expire | correct
    trade_id: str
    desk: str
    pair: str
    instrument: str               # option | hedge | spot
    notional: float
    client: str | None = None
    ref_event_id: str | None = None   # for cancel/correct: the event restated
    note: str = ""


@dataclass
class Snapshot:
    snapshot_id: str
    day: int
    desk_var: dict[str, float]
    pair_dod_pct: dict[str, float]
    positions: list[dict] = field(default_factory=list)

    @property
    def hash(self) -> str:
        blob = f"{self.snapshot_id}|{sorted(self.desk_var.items())}|{sorted(self.pair_dod_pct.items())}"
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class AtomWorld:
    seed: int
    n_days: int
    snapshots: dict[int, Snapshot] = field(default_factory=dict)   # day -> snapshot
    events: list[TradeEvent] = field(default_factory=list)
    news: list[dict] = field(default_factory=list)                 # {day, headline, body}

    def snapshot(self, day: int) -> Snapshot:
        return self.snapshots[day]

    def events_on(self, day: int) -> list[TradeEvent]:
        return [e for e in self.events if e.day == day]

    def to_dict(self) -> dict:
        return {
            "seed": self.seed, "n_days": self.n_days,
            "snapshots": {d: asdict(s) for d, s in self.snapshots.items()},
            "events": [asdict(e) for e in self.events],
            "news": self.news,
        }


def build_world(seed: int = 0, n_days: int = 30) -> AtomWorld:
    rng = random.Random(seed)
    world = AtomWorld(seed=seed, n_days=n_days)
    eid = iter(range(1, 10_000))

    open_trades: list[TradeEvent] = []
    for day in range(n_days):
        # a few lifecycle events per day
        for _ in range(rng.randint(1, 3)):
            kind = rng.choices(
                ["open", "amend", "cancel", "expire"], weights=[5, 2, 1, 1]
            )[0]
            if kind == "open" or not open_trades:
                ev = TradeEvent(
                    event_id=f"E{next(eid):05d}", day=day, kind="open",
                    trade_id=f"T{rng.randint(10_000, 99_999)}",
                    desk=rng.choice(list(DESKS)), pair=rng.choice(PAIRS),
                    instrument=rng.choice(["option", "hedge", "spot"]),
                    notional=round(rng.uniform(1, 50), 1) * 1e6,
                    client=rng.choice(CLIENTS),
                )
                open_trades.append(ev)
            else:
                src = rng.choice(open_trades)
                ev = TradeEvent(
                    event_id=f"E{next(eid):05d}", day=day, kind=kind,
                    trade_id=src.trade_id, desk=src.desk, pair=src.pair,
                    instrument=src.instrument, notional=src.notional,
                    client=src.client, ref_event_id=src.event_id,
                )
                if kind in ("cancel", "expire"):
                    open_trades.remove(src)
            world.events.append(ev)

        world.snapshots[day] = Snapshot(
            snapshot_id=f"S{seed:02d}D{day:03d}",
            day=day,
            desk_var={d: round(rng.uniform(0.5, 12.0), 2) for d in DESKS},
            pair_dod_pct={p: round(rng.gauss(0, 0.8), 3) for p in PAIRS},
            positions=[
                {"trade_id": t.trade_id, "desk": t.desk, "pair": t.pair,
                 "instrument": t.instrument, "notional": t.notional}
                for t in open_trades
            ],
        )
        if rng.random() < 0.4:
            pair = rng.choice(PAIRS)
            world.news.append({
                "day": day,
                "headline": f"{pair} volatility {'spikes' if rng.random() < 0.5 else 'eases'} "
                            f"on macro data",
                "body": f"Desks reported {'elevated' if rng.random() < 0.5 else 'muted'} "
                        f"flows in {pair}.",
            })
    return world
