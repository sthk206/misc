"""Try: vdb -m pytest examples/test_shapes.py   (click a line in the test, then Continue)"""
from shapes import Item, build_chain, total


def test_total():
    items = [Item("a", 1.0), Item("b", 2.5)]
    assert total(items) == 3.5


def test_chain():
    head = build_chain([1, 2])
    assert head.next.value == 2
