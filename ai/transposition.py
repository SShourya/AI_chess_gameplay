"""Transposition table.

Different move orders often reach the *same* position (1.Nf3 Nf6 2.Nc3 and
1.Nc3 Nf6 2.Nf3, for example).  Without a cache the search would evaluate that
position once per route.  The table remembers what the search already learned
about a position so it can be reused.

Position key
------------
The key is the board's 64-bit **Zobrist hash** (``board.hash``).  It is an
integer built by XORing one random 64-bit number for every (piece, square),
plus numbers for the side to move, castling rights and en-passant file.  It is
updated incrementally by ``make_move``/``undo_move``, so producing the key is
free, and the chance of two different positions colliding is about 1 in 2^64.

Bound flags
-----------
Alpha-beta usually does *not* find an exact score for a node: after a cutoff it
only knows the score is "at least" (lower bound) or "at most" (upper bound)
some value.  Each entry therefore stores a flag saying which of the three it
is, otherwise reusing it could return wrong results.
"""

from __future__ import annotations

from typing import Optional, Tuple

EXACT = 0   # score is the true minimax value
LOWER = 1   # score is a lower bound (a beta cutoff happened: value >= score)
UPPER = 2   # score is an upper bound (no move improved alpha: value <= score)

#: (depth, score, flag, best_move_key)
Entry = Tuple[int, float, int, Optional[tuple]]


class TranspositionTable:
    """A simple dictionary-backed table with a size cap."""

    def __init__(self, max_entries: int = 500_000) -> None:
        self.max_entries = max_entries
        self.table: dict = {}
        self.lookups = 0
        self.hits = 0
        self.stores = 0

    def probe(self, key: int) -> Optional[Entry]:
        """Return the stored entry for ``key`` or ``None``."""
        self.lookups += 1
        entry = self.table.get(key)
        if entry is not None:
            self.hits += 1
        return entry

    def store(self, key: int, depth: int, score: float, flag: int,
              best_move_key: Optional[tuple]) -> None:
        """Store an entry, keeping the deeper one if the slot is occupied."""
        old = self.table.get(key)
        if old is not None and old[0] > depth:
            return
        if len(self.table) >= self.max_entries and key not in self.table:
            self.table.clear()           # crude but effective replacement scheme
        self.table[key] = (depth, score, flag, best_move_key)
        self.stores += 1

    def clear(self) -> None:
        self.table.clear()
        self.lookups = self.hits = self.stores = 0

    def __len__(self) -> int:
        return len(self.table)
