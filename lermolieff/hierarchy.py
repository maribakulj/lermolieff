"""The Iconclass hierarchy read from the notation itself, without the official tree.

A notation is one or more parts joined by ':'. A part is read as tokens: each of the first two digits, then a letter or a doubled
letter, then each digit, then any bracket (a name '(NAME)' or a key '(+..)') as one token, and the digits after it one by one.
Its ancestors are the prefixes ending on a token. A compound keeps its own string and the ancestors of each part.
"""

from __future__ import annotations

import re

PART = re.compile(r"\d\d?[A-Z]{0,2}\d*(?:\([^)]*\)\d*)*")
TOKEN = re.compile(r"\([^)]*\)|[A-Z]{1,2}|\d")
_cache: dict[str, frozenset[str]] = {}


def ancestors(code: str) -> frozenset[str]:
    """The code and all its ancestors; an unreadable code is itself only."""
    code = code.strip()
    if code in _cache:
        return _cache[code]
    parts = code.split(":")
    out = {code}
    if all(PART.fullmatch(p) for p in parts):
        for p in parts:
            cur = ""
            for t in TOKEN.findall(p):
                cur += t
                out.add(cur)
    _cache[code] = frozenset(out)
    return _cache[code]


def readable(code: str) -> bool:
    return all(PART.fullmatch(p) for p in code.strip().split(":"))


def most_specific(codes: list[str]) -> list[str]:
    """Drop every code implied by another one of the list (its ancestors); order kept."""
    codes = list(dict.fromkeys(codes))
    implied: set[str] = set()
    for c in codes:
        implied |= ancestors(c) - {c}
    return [c for c in codes if c not in implied]


def terms(code: str) -> list[str]:
    """The parts of a compound notation, split on the colons outside brackets only."""
    return [t.strip() for t in re.split(r":(?![^(]*\))", code) if t.strip()]
