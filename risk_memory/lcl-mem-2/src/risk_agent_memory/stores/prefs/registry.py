"""Preference-key registry: type/allowed-value validation per B.1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class UnknownPrefKey(KeyError):
    pass


class InvalidPrefValue(ValueError):
    pass


class PrefRegistry:
    def __init__(self, spec: dict[str, Any]):
        self.version = spec.get("version", 1)
        self.keys: dict[str, dict] = spec["keys"]

    @classmethod
    def load(cls, path: str | Path) -> "PrefRegistry":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    def consumer_of(self, key: str) -> str:
        return self.spec_for(key).get("consumer", "unknown")

    def spec_for(self, key: str) -> dict:
        if key not in self.keys:
            raise UnknownPrefKey(
                f"unknown preference key {key!r}; registered keys: {sorted(self.keys)}"
            )
        return self.keys[key]

    def validate(self, key: str, value: Any) -> Any:
        """Raises UnknownPrefKey / InvalidPrefValue; returns the value."""
        spec = self.spec_for(key)
        t = spec["type"]
        if t == "enum":
            if value not in spec["allowed"]:
                raise InvalidPrefValue(
                    f"{key}: {value!r} not in allowed {spec['allowed']}"
                )
        elif t == "bool":
            if not isinstance(value, bool):
                raise InvalidPrefValue(f"{key}: expected bool, got {type(value).__name__}")
        elif t == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InvalidPrefValue(f"{key}: expected number, got {type(value).__name__}")
            if "min" in spec and value < spec["min"]:
                raise InvalidPrefValue(f"{key}: {value} < min {spec['min']}")
            if "max" in spec and value > spec["max"]:
                raise InvalidPrefValue(f"{key}: {value} > max {spec['max']}")
        elif t == "list":
            if not isinstance(value, list):
                raise InvalidPrefValue(f"{key}: expected list, got {type(value).__name__}")
            allowed = spec.get("item_allowed")
            if allowed:
                bad = [v for v in value if v not in allowed]
                if bad:
                    raise InvalidPrefValue(f"{key}: items {bad} not in allowed {allowed}")
        elif t == "str":
            if not isinstance(value, str):
                raise InvalidPrefValue(f"{key}: expected str, got {type(value).__name__}")
        else:
            raise InvalidPrefValue(f"{key}: registry has unknown type {t!r}")
        return value
