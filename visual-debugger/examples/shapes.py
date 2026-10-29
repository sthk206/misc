"""A small program that exercises every shape vdb draws. Try: vdb examples/shapes.py"""
from dataclasses import dataclass, field


@dataclass
class Item:
    name: str
    price: float
    tags: set = field(default_factory=set)


class Node:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next


def build_chain(values):
    head = None
    for v in reversed(values):
        head = Node(v, head)
    return head


def total(items):
    running = 0.0
    for item in items:
        running += item.price
    return running


def main():
    items = [Item("apple", 0.5, {"fruit"}), Item("bread", 2.25)]
    cart = {"owner": "Ana", "items": items, "coords": (3, 4)}
    alias = items  # the same list under two names: watch both arrows
    alias.append(Item("milk", 1.2, {"dairy", "cold"}))
    chain = build_chain([1, 2, 3])
    print("total", total(cart["items"]))
    print("chain starts at", chain.value)


if __name__ == "__main__":
    main()
