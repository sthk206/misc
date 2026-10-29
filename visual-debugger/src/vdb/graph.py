"""Turn the live objects reachable from paused frames into a JSON graph.

Every heap object becomes a node keyed by its id(); every reference to it
becomes {"r": id}. Scalars (numbers, strings, enums, dates, functions...) are
drawn inside the box that holds them, as {"p": text, "t": type}.
"""

import argparse
import dataclasses
import datetime
import decimal
import enum
import fractions
import itertools
import pathlib
import sys
import types
import uuid
from collections import deque

MAX_OBJECTS = 300
MAX_ITEMS = 40
MAX_STR = 80
MAX_TABLE_ROWS = 15
MAX_TABLE_COLS = 12

VALUE_TYPES = (
    datetime.date, datetime.time, datetime.timedelta, decimal.Decimal,
    fractions.Fraction, uuid.UUID, pathlib.PurePath, range,
)
CODE_TYPES = (
    types.FunctionType, types.BuiltinFunctionType, types.MethodType,
    types.ModuleType, type,
)
HIDDEN_AT_MODULE_LEVEL = (types.ModuleType, type, types.FunctionType, types.BuiltinFunctionType)


def safe_repr(obj, limit=MAX_STR):
    try:
        if isinstance(obj, str) and len(obj) > limit:
            text = repr(obj[:limit])
        else:
            text = repr(obj)
    except Exception as exc:
        return f"<{type(obj).__name__}: repr failed ({type(exc).__name__})>"
    return text if len(text) <= limit else text[:limit - 1] + "…"


def safe_str(obj, limit=40):
    try:
        text = str(obj)
    except Exception:
        return safe_repr(obj, limit)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def code_label(obj):
    if isinstance(obj, types.ModuleType):
        return f"module {obj.__name__}"
    if isinstance(obj, type):
        return f"class {obj.__qualname__}"
    if isinstance(obj, types.MethodType):
        return f"method {getattr(obj.__func__, '__qualname__', '?')}"
    if isinstance(obj, types.BuiltinFunctionType):
        return f"builtin {obj.__name__}"
    return f"function {getattr(obj, '__qualname__', '?')}"


def scalar(obj):
    """The inline form of obj, or None when obj deserves its own box."""
    if obj is None:
        return {"p": "None", "t": "none"}
    if isinstance(obj, enum.Enum):
        return {"p": f"{type(obj).__name__}.{obj.name}", "t": "enum"}
    if isinstance(obj, bool):
        return {"p": repr(obj), "t": "bool"}
    if isinstance(obj, (int, float, complex)):
        return {"p": safe_repr(obj), "t": "num"}
    if isinstance(obj, str):
        return {"p": safe_repr(obj), "t": "str"}
    if isinstance(obj, (bytes, bytearray)):
        return {"p": safe_repr(obj), "t": "bytes"}
    if isinstance(obj, VALUE_TYPES):
        return {"p": safe_repr(obj), "t": "value"}
    if isinstance(obj, CODE_TYPES):
        return {"p": code_label(obj), "t": "code"}
    if type(obj).__module__ == "numpy" and getattr(obj, "ndim", None) == 0:
        return {"p": safe_str(obj), "t": "num"}
    return None


def library_kind(t):
    """'dataframe' / 'series' / 'array' for pandas and numpy containers."""
    for cls in t.__mro__:
        module = cls.__module__ or ""
        if module.startswith("pandas") and cls.__name__ in ("DataFrame", "Series"):
            return cls.__name__.lower()
        if module == "numpy" and cls.__name__ == "ndarray":
            return "array"
    return None


def is_pydantic(t):
    return any((cls.__module__ or "").startswith("pydantic") for cls in t.__mro__[1:])


def is_namedtuple(obj):
    return isinstance(obj, tuple) and hasattr(type(obj), "_fields")


def visible_locals(frame):
    """The frame's variables, minus dunders and, at module level, imports and definitions."""
    at_module_level = frame.f_code.co_name == "<module>"
    for name, value in list(frame.f_locals.items()):
        if name.startswith(("__", ".", "@")):
            continue
        if at_module_level and (
            isinstance(value, HIDDEN_AT_MODULE_LEVEL)
            or type(value).__module__ in ("typing", "typing_extensions")
        ):
            continue
        yield name, value


class GraphBuilder:
    """Builds one snapshot. is_user_file decides which classes get expanded."""

    def __init__(self, is_user_file):
        self.is_user_file = is_user_file
        self.objects = {}
        self._pending = deque()
        self._user_class = {}

    def build(self, frames, hidden_outer=0):
        """frames: (frame, lineno, hidden_below) tuples, innermost first."""
        out = []
        for frame, lineno, hidden_below in frames:
            code = frame.f_code
            name = code.co_name
            if name == "<module>":
                name = f"module {frame.f_globals.get('__name__', '?')}"
            out.append({
                "name": name,
                "file": code.co_filename,
                "line": lineno,
                "hidden_below": hidden_below,
                "locals": [[n, self.value(v)] for n, v in visible_locals(frame)],
            })
        while self._pending:
            oid, obj = self._pending.popleft()
            self.objects[oid] = self.describe(obj)
        return {"frames": out, "objects": self.objects, "hidden_outer": hidden_outer}

    def value(self, obj):
        inline = scalar(obj)
        if inline is not None:
            return inline
        oid = str(id(obj))
        if oid not in self.objects:
            if len(self.objects) >= MAX_OBJECTS:
                return {"p": f"{type(obj).__name__} … (graph limit)", "t": "more"}
            self.objects[oid] = None
            self._pending.append((oid, obj))
        return {"r": oid}

    def describe(self, obj):
        t = type(obj)
        try:
            return self._describe(obj, t)
        except Exception as exc:
            return {"kind": "opaque", "type": t.__name__, "text": f"could not inspect: {exc!r}"}

    def _describe(self, obj, t):
        name = t.__name__
        if isinstance(obj, dict):
            entries = [[self.value(k), self.value(v)] for k, v in itertools.islice(obj.items(), MAX_ITEMS)]
            return {"kind": "dict", "type": name, "entries": entries, "len": len(obj)}
        if is_namedtuple(obj):
            attrs = [[f, self.value(v)] for f, v in zip(t._fields, obj)]
            return {"kind": "instance", "type": name, "badge": "namedtuple", "attrs": attrs}
        if isinstance(obj, (list, tuple, deque)):
            kind = "tuple" if isinstance(obj, tuple) else "list"
            return self._sequence(kind, name, obj)
        if isinstance(obj, (set, frozenset)):
            try:
                ordered = sorted(obj)
            except TypeError:
                ordered = list(obj)
            return self._sequence("set", name, ordered, len(obj))
        kind = library_kind(t)
        if kind:
            return getattr(self, "_" + kind)(obj, name)
        if dataclasses.is_dataclass(obj):
            attrs = [[f.name, self.value(getattr(obj, f.name, None))] for f in dataclasses.fields(obj)]
            return {"kind": "instance", "type": name, "badge": "dataclass", "attrs": attrs}
        if is_pydantic(t) and isinstance(getattr(obj, "__dict__", None), dict):
            attrs = [[k, self.value(v)] for k, v in obj.__dict__.items()]
            return {"kind": "instance", "type": name, "badge": "pydantic", "attrs": attrs}
        if self._is_user_class(t) or isinstance(obj, (argparse.Namespace, types.SimpleNamespace)):
            return {"kind": "instance", "type": name, "attrs": self._instance_attrs(obj, t)}
        return {"kind": "opaque", "type": name, "text": safe_repr(obj, 160)}

    def _sequence(self, kind, name, items, length=None):
        values = [self.value(v) for v in itertools.islice(items, MAX_ITEMS)]
        return {"kind": kind, "type": name, "items": values, "len": len(items) if length is None else length}

    def _instance_attrs(self, obj, t):
        attrs = []
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            attrs.extend(d.items())
        for cls in t.__mro__:
            slots = cls.__dict__.get("__slots__", ())
            for slot in (slots,) if isinstance(slots, str) else slots:
                if slot in ("__dict__", "__weakref__"):
                    continue
                try:
                    attrs.append((slot, getattr(obj, slot)))
                except AttributeError:
                    continue
        return [[k, self.value(v)] for k, v in attrs[:MAX_ITEMS]]

    def _is_user_class(self, t):
        known = self._user_class.get(t)
        if known is None:
            module = sys.modules.get(t.__module__)
            filename = getattr(module, "__file__", None)
            known = self._user_class[t] = bool(filename) and self.is_user_file(filename)
        return known

    def _dataframe(self, df, name):
        part = df.iloc[:MAX_TABLE_ROWS, :MAX_TABLE_COLS]
        return {
            "kind": "dataframe", "type": name, "shape": list(df.shape),
            "columns": [safe_str(c) for c in part.columns],
            "dtypes": [str(d) for d in part.dtypes],
            "index": [safe_str(i) for i in part.index],
            "rows": [[safe_str(v) for v in row] for row in part.itertuples(index=False, name=None)],
        }

    def _series(self, s, name):
        part = s.iloc[:MAX_TABLE_ROWS]
        return {
            "kind": "series", "type": name, "name": safe_str(s.name), "dtype": str(s.dtype),
            "len": len(s), "index": [safe_str(i) for i in part.index],
            "values": [safe_str(v) for v in part.tolist()],
        }

    def _array(self, a, name):
        out = {"kind": "array", "type": name, "dtype": str(a.dtype), "shape": list(a.shape), "len": int(a.size)}
        if a.ndim == 1:
            out["items"] = [safe_str(v) for v in a[:MAX_ITEMS].tolist()]
        elif a.ndim == 2:
            out["rows"] = [[safe_str(v) for v in row] for row in a[:MAX_TABLE_ROWS, :MAX_TABLE_COLS].tolist()]
        else:
            out["text"] = safe_repr(a, 300)
        return out
