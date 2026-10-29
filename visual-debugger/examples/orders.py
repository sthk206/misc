"""pandas + pydantic + dataclasses. Needs pandas and pydantic in the venv. Try: vdb examples/orders.py"""
from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel


class Order(BaseModel):
    id: int
    customer: str
    amount: float


@dataclass
class Summary:
    customer: str
    total: float
    orders: int


def summarize(orders):
    df = pd.DataFrame([o.model_dump() for o in orders])
    totals = df.groupby("customer")["amount"].sum()
    counts = df["customer"].value_counts()
    return [Summary(name, float(totals[name]), int(counts[name])) for name in totals.index]


def main():
    orders = [
        Order(id=1, customer="ana", amount=12.5),
        Order(id=2, customer="ben", amount=7.0),
        Order(id=3, customer="ana", amount=3.25),
    ]
    summaries = summarize(orders)
    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()
