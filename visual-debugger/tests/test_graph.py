import enum
import os
import sys
import unittest
from collections import namedtuple
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vdb.graph import MAX_ITEMS, GraphBuilder, safe_repr  # noqa: E402

Point = namedtuple("Point", "x y")


class Color(enum.Enum):
    RED = 1


@dataclass
class Box:
    label: str
    contents: list


class Plain:
    def __init__(self):
        self.a = 1


class BadRepr:
    __slots__ = ()

    def __repr__(self):
        raise RuntimeError("boom")


class GraphTest(unittest.TestCase):
    def build(self, value):
        builder = GraphBuilder(lambda f: os.path.realpath(f) == os.path.realpath(__file__))
        ref = builder.value(value)
        builder.build([])
        return ref, builder.objects

    def test_scalars_are_inline(self):
        for value, kind in [(3, "num"), ("hi", "str"), (None, "none"), (True, "bool"), (Color.RED, "enum"), (len, "code")]:
            ref, objects = self.build(value)
            self.assertEqual(ref["t"], kind)
            self.assertEqual(objects, {})

    def test_shared_reference_is_one_node(self):
        shared = [1, 2]
        ref, objects = self.build({"a": shared, "b": shared})
        entries = objects[ref["r"]]["entries"]
        self.assertEqual(entries[0][1], entries[1][1])
        self.assertEqual(len(objects), 2)

    def test_cycle_terminates(self):
        loop = []
        loop.append(loop)
        ref, objects = self.build(loop)
        self.assertEqual(objects[ref["r"]]["items"][0], ref)

    def test_kinds(self):
        cases = [
            ([1], "list"), ((1,), "tuple"), ({1}, "set"), ({1: 2}, "dict"),
            (Point(1, 2), "instance"), (Box("x", []), "instance"), (Plain(), "instance"),
        ]
        for value, kind in cases:
            ref, objects = self.build(value)
            self.assertEqual(objects[ref["r"]]["kind"], kind, value)

    def test_namespaces_expand(self):
        import argparse
        ref, objects = self.build(argparse.Namespace(cmd="validate", out="results"))
        self.assertEqual(objects[ref["r"]]["attrs"], [["cmd", {"p": "'validate'", "t": "str"}], ["out", {"p": "'results'", "t": "str"}]])

    def test_badges(self):
        ref, objects = self.build(Box("x", [1]))
        self.assertEqual(objects[ref["r"]]["badge"], "dataclass")
        ref, objects = self.build(Point(1, 2))
        self.assertEqual(objects[ref["r"]]["badge"], "namedtuple")

    def test_long_list_is_truncated(self):
        ref, objects = self.build(list(range(1000)))
        node = objects[ref["r"]]
        self.assertEqual(len(node["items"]), MAX_ITEMS)
        self.assertEqual(node["len"], 1000)

    def test_broken_repr_does_not_crash(self):
        self.assertIn("repr failed", safe_repr(BadRepr()))

    def test_library_instances_stay_collapsed(self):
        import logging
        ref, objects = self.build(logging.getLogger("x"))
        self.assertEqual(objects[ref["r"]]["kind"], "opaque")


if __name__ == "__main__":
    unittest.main()
