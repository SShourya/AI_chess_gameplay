# Chess AI — Python + Pygame

A complete, from-scratch chess engine and GUI: real move generation (no
`python-chess`), a Minimax + Alpha-Beta search (no Stockfish), and a Pygame
board you can actually play on.

```bash
pip install -r requirements.txt
python main.py
```

You play **White** by clicking a piece then a highlighted destination square.
The computer plays **Black** and thinks on a background thread, so the window
never freezes while it searches.

```
python main.py --difficulty Hard      # Easy (depth 2) | Medium (depth 3) | Hard (depth 4)
python main.py --depth 5              # override the depth directly
python main.py --debug                # print search stats to the console after every AI move
python main.py --vector-pieces        # force the hand-drawn piece fallback (see "Piece rendering" below)
```

Keys: `R` / `N` restart · `Q` quit · `Esc` cancel a selection or the promotion popup.

## 1. Features

- Full chess rules: every piece's movement, castling (both sides, all five
  legality conditions), en passant, and pawn promotion with a piece-choice
  popup.
- Check, checkmate and stalemate detection, plus draw detection (insufficient
  material, the fifty-move rule, threefold repetition).
- A real Minimax search with Alpha-Beta pruning, move ordering (MVV-LVA +
  checks + a transposition-table hint), a transposition table, and check
  extensions so the AI doesn't miss forced mates just past the horizon.
- A positional evaluation function: material, piece-square tables, centre
  control, development, king safety/castling, pawn structure (doubled /
  isolated / passed pawns), mobility, and an endgame "mop-up" term that drives
  a lone king to the edge when one side is clearly winning.
- Difficulty selector (Easy/Medium/Hard = depth 2/3/4) that takes effect on
  the computer's next move.
- A modern side panel: turn indicator, check/checkmate/stalemate/draw status,
  captured pieces with a material-lead badge, live AI statistics (nodes
  searched, nodes pruned, evaluation, time), a scrolling move list, and
  restart/quit buttons.
- Selected-square highlight, legal-move dots (rings on empty squares, filled
  circles on captures — including en passant), last-move highlight, and a
  pulsing red glow on a king in check.
- A resizable window: the board recomputes its layout every frame and stays
  square from small laptop sizes up through 1920x1080 and beyond.
- Unicode chess glyphs when a font that actually contains them is available,
  with a hand-drawn polygon piece set as an automatic fallback — the game
  never shows blank squares or "tofu" boxes (see below).

## 2. Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Only dependency: `pygame`. No `python-chess`, no Stockfish, no other engine.

## 3. How to play

1. Run `python main.py`. You are White; the board starts in the normal
   position.
2. Click one of your pieces — its legal destinations light up.
3. Click a highlighted square to move there. Clicking a different one of your
   own pieces instead just changes the selection; clicking anywhere else
   clears it.
4. If the move is a pawn reaching the last rank, a popup asks which piece to
   promote to.
5. The panel switches to "Computer is thinking…" and the board is briefly
   locked while the AI searches on a background thread.
6. When the game ends, a dialog reports checkmate/stalemate/draw with
   **Play Again** / **Quit** buttons.

## 4. Project structure

```
python-chess-ai/
├── main.py                  # app entry point: the Pygame loop, wires everything together
├── requirements.txt
├── README.md
│
├── game/
│   ├── pieces.py             # piece codes, values, movement offset tables
│   ├── board.py               # Board: grid, FEN import/export, Zobrist hashing
│   ├── move.py                 # Move class + make_move / undo_move
│   ├── rules.py                  # attack detection, pseudo-legal + legal move generation, perft
│   └── game_state.py              # turn handling, status, captured pieces, move log — the UI's only gateway to the board
│
├── ai/
│   ├── evaluation.py          # evaluate_board(): material + piece-square tables + positional terms
│   ├── minimax.py               # minimax(), alpha_beta(), move ordering, find_best_move()
│   ├── transposition.py           # Zobrist-hash-keyed transposition table
│   └── worker.py                    # runs find_best_move on a background thread
│
├── ui/
│   ├── board_renderer.py      # board/piece drawing, responsive layout, promotion popup
│   ├── menu.py                  # buttons, difficulty selector, side panel, game-over dialog
│   └── input_handler.py           # turns mouse/keyboard events into GameState.play() calls
│
└── tests/
    ├── test_piece_movement.py   # per-piece movement rules
    ├── test_special_rules.py    # castling, en passant, check/mate/stalemate, draws
    ├── test_engine.py           # perft vs. published counts, make/undo, Zobrist hashing, FEN
    ├── test_ai.py                # evaluation, Minimax == Alpha-Beta, pruning, tactics, the worker thread
    └── test_ui.py                  # headless Pygame tests: clicking, promotion, resize, a full random game
```

Every legal move — human or AI — is played through **one** function,
`GameState.play(from_square, to_square, promotion)`. It looks the move up in
the legal-move list `game_state._refresh_status()` just computed and only
plays it if found, which is what makes "the AI can never play an illegal
move" an actual guarantee rather than a hope: `main.py`'s `_apply_ai_result`
calls the exact same gate the human's clicks go through.

## 5. Board representation

`Board.grid` is an 8x8 list of lists of piece codes or `None`. A piece code
is two characters, colour + type — `"wp"` = white pawn, `"bk"` = black king —
chosen over piece objects because strings are immutable, hashable, and cheap
to print for debugging. `grid[0]` is rank 8 (Black's back rank) and `grid[7]`
is rank 1, so the array literally looks like a chess diagram:

```python
grid[0] = ["br", "bn", "bb", "bq", "bk", "bb", "bn", "br"]   # rank 8
grid[1] = ["bp"] * 8                                          # rank 7
...
grid[6] = ["wp"] * 8                                          # rank 2
grid[7] = ["wr", "wn", "wb", "wq", "wk", "wb", "wn", "wr"]   # rank 1
```

Alongside the grid, `Board` tracks side to move, castling rights (a 4-bit
mask), the en-passant target square, the half-move clock (fifty-move rule)
and the full-move number — i.e. everything a FEN string encodes.
`Board.from_fen` / `Board.to_fen` convert both ways, and `Board.copy()` makes
an independent deep copy (used to hand a position to the AI's worker thread
without any locking).

## 6. Move generation

`generate_pseudo_legal_moves(board, color)` generates every move that obeys
each piece's movement pattern — including castling (with its own "not
through/out of check" checks, since those can't be verified by the
make-and-test step below) and en passant — but may leave the mover's own king
in check.

`generate_legal_moves(board, color)` filters that list down: for every
pseudo-legal move it calls `make_move`, asks `is_in_check` whether the
mover's own king is now attacked, and calls `undo_move` — keeping the move
only if the king was safe. This "generate, test, undo" approach is simple to
get right and is exactly what the engine's own tests (and the perft
benchmark below) exist to make sure it *is* right — the AI and the UI never
touch the pseudo-legal list directly.

`is_square_attacked(board, square, attacking_color)` is the shared primitive
underneath check detection, castling validation and the legality filter
above: instead of generating every attacker's moves, it looks *outward* from
the target square (pawn attack squares, knight-move squares, adjacent king
squares, and sliding rays for bishop/rook/queen) and stops at the first piece
found in each direction.

### Correctness: perft

[Perft](https://www.chessprogramming.org/Perft) counts leaf nodes of the
legal-move tree to a given depth; a wrong move generator almost always
produces a wrong count somewhere. `tests/test_engine.py` checks the standard
position plus the four widely-used "Kiwipete"/positions 3–5 (which
specifically stress castling, en passant, promotions and pins) against their
published values through depth 3–4, e.g.:

```
start position   depth 4: 197,281   (matches)
Kiwipete          depth 3:  97,862   (matches)
```

## 7. Minimax

`minimax(board, depth, maximizing_player)` is the textbook, unpruned
algorithm: at the root, the AI's side maximises its score; the opponent's
side minimises it; every legal move at every node is explored down to depth
0, where `evaluate_board` scores the position. It exists mainly as a
correctness reference — `tests/test_ai.py` asserts `alpha_beta` returns the
*exact same value* as `minimax` on the same position at the same depth, for
several positions and both colours to move.

## 8. Alpha-Beta pruning

`alpha_beta(board, depth, alpha, beta, maximizing_player)` searches the same
tree but skips branches that provably cannot change the result:

- `alpha` = the best score the maximiser can already guarantee somewhere on
  the path to this node.
- `beta` = the best score the minimiser can already guarantee.
- If at any point `alpha >= beta`, the side to move at the node *above* this
  one already has a better alternative than letting the game reach here, so
  the remaining sibling moves are skipped — that's the cutoff, marked with an
  explicit `# <-- BETA CUTOFF` / `# <-- ALPHA CUTOFF` comment at the two
  `break` statements in `ai/minimax.py`.

`SearchStats` (returned as part of every `find_best_move` result) counts
`nodes`, `cutoffs` and `pruned` moves, so the panel's "Nodes searched / Nodes
pruned" line is real, measured data, not an estimate. In `tests/test_ai.py`,
`test_pruning_actually_reduces_the_search` checks pruning cuts the node count
by more than 3x on a tactical middlegame position at depth 2.

## 9. Evaluation function

`ai/evaluation.py`'s `evaluate_board(board, perspective)` returns a score in
centipawns (100 = one pawn) from `perspective`'s point of view. It combines:

- **Material** — pawn 100, knight 320, bishop 330, rook 500, queen 900, king
  20000.
- **Piece-square tables** — Michniewski's well-known "simplified evaluation"
  tables for every piece, with a separate king table for the middlegame
  (stay castled, stay safe) versus the endgame (get active, help mate).
- **Centre control** — a small bonus for pawns/knights/bishops on d4/e4/d5/e5.
- **Development** — a penalty for minor pieces still sitting on their home
  squares in the middlegame.
- **King safety** — bonuses for pawns still shielding the king, a penalty for
  an open file next to it, and a `CASTLED_BONUS` for having actually castled
  (with a smaller penalty for having *lost* the right to, without having
  used it).
- **Pawn structure** — penalties for doubled and isolated pawns, and a
  bonus for passed pawns that grows sharply as the pawn advances (and is
  doubled again in the endgame).
- **Mobility** — knights/bishops/rooks/queens score a small bonus per legal
  square they currently reach.
- **Check** — a small penalty for the side currently in check.
- **Endgame mop-up** — once one side is clearly ahead in material with no
  queen-heavy complications, a term rewards driving the enemy king toward
  the edge/corner and the winning king toward it, which is what lets the AI
  actually convert a won K+Q vs K ending instead of shuffling forever (see
  `test_ai_does_not_stalemate_when_winning`).

## 10. Checkmate scoring, piece-square tables & move ordering

**Checkmate** is worth `CHECKMATE_SCORE = 1_000_000`, minus the number of
plies (`ply`) it took to reach it (`ai/evaluation.py`'s `terminal_score`, also
used inline in `minimax`/`alpha_beta`). That small ply-dependent adjustment
is what makes the AI prefer a mate in 1 over a mate in 3, and — symmetrically
— prefer to delay being mated as long as possible rather than resign itself
to the fastest loss. `format_score` turns a score like `999999` back into the
human-readable `"Mate in 1"` shown by `--debug`.

**Piece-square tables** are described above; they live in
`ai/evaluation.py` as flat 8x8 tables written from White's perspective, with
Black's version built once at import time by vertically mirroring them.

**Move ordering** (`order_moves` in `ai/minimax.py`) sorts each node's moves
before searching them, since Alpha-Beta prunes far more when the best move is
tried first: checkmating moves first, then the transposition table's
remembered best move, then captures ranked by MVV-LVA (Most Valuable Victim,
Least Valuable Attacker — a pawn taking a queen is tried long before a queen
taking a pawn), then promotions, then other checks, then everything else.
`test_move_ordering_reduces_nodes` and `test_mvv_lva_prefers_cheap_attacker`
in `tests/test_ai.py` check this concretely rather than just asserting it
exists.

## 11. Transposition table

`ai/transposition.py`'s `TranspositionTable` caches what the search already
learned about a position, keyed by the position's **Zobrist hash**
(`board.hash`): one random 64-bit number is assigned to every (piece, square)
pair at start-up, plus numbers for the side to move, each castling-rights
combination, and each en-passant file; a position's hash is the XOR of every
number that applies to it. `make_move`/`undo_move` update the hash
incrementally (a handful of XORs) rather than recomputing it from all 64
squares, so hashing is effectively free — and because XOR is its own
inverse, `undo_move` restores the exact pre-move hash.

Since Alpha-Beta usually only proves a *bound* on a node's value rather than
its exact value, each entry stores a flag (`EXACT` / `LOWER` / `UPPER`)
alongside the depth and score, and a stored value is only trusted when it was
computed at least as deep as the current query.

## 12. Difficulty levels & keeping the UI responsive

Easy/Medium/Hard map to depth 2/3/4 (`DIFFICULTY_DEPTHS` in `ai/minimax.py`).
Since a depth-4 search can take real time, `ai/worker.py`'s `AIWorker` runs
`find_best_move` on a **background thread** against a private `board.copy()`
— the UI thread keeps drawing and responding to the difficulty selector and
window resizes at 60 FPS while the search runs, and `main.py` polls
`worker.poll()` once per frame rather than blocking. Restarting mid-search
cancels the in-flight job via a `threading.Event` the search checks
periodically, so a stale result can never be applied to the new game
(`test_restart_during_ai_search_discards_the_stale_move`).

## 13. Piece rendering — no missing-font crashes

`ui/board_renderer.py`'s `PieceRenderer` looks for an installed font that
*actually contains* the six chess glyphs (checked with `Font.metrics()`,
which returns `None` per-character for glyphs a font can't render — this is
what catches the common case of a font that "exists" but only has empty
"tofu" boxes for chess symbols) across a list of common candidates
(DejaVu Sans, Segoe UI Symbol, Apple Symbols, Noto Sans Symbols, …). If one is
found, pieces are drawn as outlined Unicode glyphs; if none is found — or if
`CHESS_VECTOR_PIECES=1` is set, or `main.py` is run with `--vector-pieces` —
every piece is instead drawn as a small set of filled polygons, so the game
always shows recognisable pieces, never blank squares. The general UI font
(labels, buttons, the panel) separately falls back through a chain of common
system font names and finally to `pygame.font.get_default_font()`, so a
missing font can never crash the game anywhere in the UI.

## 14. Testing

```bash
python -m unittest discover -s tests      # ~14s, 141 tests
CHESS_SLOW_TESTS=1 python -m unittest discover -s tests   # also runs the depth-5 perft check (slow)
```

What's covered (see the file list in §4 above for exactly where):

- **Every piece type**: normal movement, blocking, capturing, edge/corner
  cases (e.g. a knight in the corner has exactly 2 moves), pawns' double
  step / diagonal-only capture / promotion-by-capture.
- **Castling**: kingside and queenside for both colours, and every way it
  can become illegal — rights already lost, pieces in the way (including the
  queenside-only b-file square), rights lost after the king/rook moves *or*
  the rook is captured, and the three check-related conditions (currently in
  check, king would pass through an attacked square, king would land in
  check) tested as separate cases.
- **En passant**: available immediately after a double push, only then (not
  after any intervening move), for both colours, and the specific case where
  taking en passant would illegally expose your own king (a pinned-pawn
  edge case many implementations miss).
- **Check**: detection from every piece type, `is_square_attacked` directly,
  forced-response move lists, pins (a pinned piece has zero legal moves;
  a pinned rook may still slide along the pin line), and double check.
- **Checkmate**: Fool's mate and Scholar's mate played move-by-move through
  `GameState`, a smothered-mate pattern, and back-rank mate via the
  lower-level `is_checkmate`.
- **Stalemate**: a known K+Q vs K stalemate position, and confirming it is
  *not* also reported as checkmate.
- **Draws**: insufficient material (K v K, K+minor v K, opposite/same-colour
  bishops), the fifty-move rule, and threefold repetition.
- **The AI**: Alpha-Beta returns the identical value to plain Minimax at
  matching depths (the main correctness guarantee), pruning measurably
  reduces node count, move ordering measurably reduces node count, the AI
  takes free material, prefers the bigger of two available captures, finds
  mate in 1 and mate in 2, provably avoids moves that walk into a mate in 1,
  and does not stalemate a won King+Queen ending. `test_ai_only_ever_plays_legal_moves_in_self_play`
  runs several full self-play games and checks every single AI move against
  the independently-computed legal-move list.
- **The engine as a whole**: perft against five published reference
  positions (through depth 4), that `undo_move` restores the position
  *exactly* (FEN, Zobrist hash, king positions, repetition history) for
  every legal move in several positions, that random 60-ply playouts never
  drift the incremental hash away from a from-scratch recomputation, and
  that transposed move orders reaching the same position hash identically.
- **The UI**, driven headlessly (SDL's "dummy" video driver, no real
  display needed): click-to-select, click-to-move, illegal destinations
  being ignored, the AI thread actually replying and updating the stats
  panel, the human being locked out while the AI thinks, the promotion
  popup (choosing a piece, and cancelling with Esc), the difficulty
  selector, restart (including cancelling a search in progress), the
  game-over dialog and Play Again, window resizing at six different sizes
  from 900×640 up to 1920×1080, and a full randomized game played through
  real mouse clicks against the real AI thread, replayed independently
  afterward to confirm every move — human and AI — was legal.

## How the AI thinks

```text
Current Position
       ↓
Generate Legal Moves            (generate_pseudo_legal_moves + the check-safety filter)
       ↓
Order Moves                     (checkmates → TT move → captures (MVV-LVA) → promotions → checks → rest)
       ↓
Minimax Search                  (maximise on the AI's nodes, minimise on the opponent's)
       ↓
Alpha-Beta Pruning              (skip branches once alpha >= beta)
       ↓
Evaluate Position                (material + piece-square tables + positional terms, at depth 0)
       ↓
Return Best Move                (find_best_move — runs on a background thread via AIWorker)
       ↓
Execute Move                     (through the same GameState.play() gate as the human's clicks)
```
