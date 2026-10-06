"""Drawing the board, the pieces and all board overlays.

Layout is computed from the window size every frame (:func:`compute_layout`)
so the board always stays square and everything resizes cleanly.

Pieces are drawn as small filled polygons by default -- crisp, identical on
every platform, and not dependent on any particular font containing chess
symbols. Unicode chess glyphs (some system fonts render these nicely) can be
opted into with ``CHESS_UNICODE_PIECES=1`` or ``--unicode-pieces``, with an
extra check to reject fonts that only have "tofu box" placeholders for them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pygame

from game.board import Square
from game.game_state import GameState
from game.pieces import PROMOTION_TYPES, UNICODE_SOLID

# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
BACKGROUND = (30, 33, 41)
COORD_COLOR = (170, 176, 190)
SELECTED_COLOR = (246, 246, 105, 150)
LAST_MOVE_COLOR = (205, 210, 106, 140)
TARGET_DOT_COLOR = (25, 25, 25, 70)
CHECK_COLOR = (230, 40, 40)

_SYMBOL_FONT_CANDIDATES = (
    "dejavusans", "segoeuisymbol", "applesymbols", "notosanssymbols2",
    "arialunicodems", "freeserif", "symbola", "notosansmono", "menlo",
)

_font_cache: Dict[Tuple[str, int, bool], pygame.font.Font] = {}


def ui_font(size: int, bold: bool = False) -> pygame.font.Font:
    """A UI font with a safe fallback chain (SysFont falls back to pygame's
    default font if none of the names exist)."""
    key = ("ui", size, bold)
    if key not in _font_cache:
        _font_cache[key] = pygame.font.SysFont("segoeui,helveticaneue,helvetica,arial,dejavusans", size, bold=bold)
    return _font_cache[key]


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
@dataclass
class Layout:
    window: Tuple[int, int]
    board_rect: pygame.Rect       # exactly the 8x8 squares
    square: int
    panel_rect: pygame.Rect


def compute_layout(width: int, height: int) -> Layout:
    """Board on the left, information panel on the right, both centred."""
    margin = 20
    coord = 26
    board_room = min(height - 2 * margin - coord, width - 300 - 3 * margin - coord)
    square = max(24, board_room // 8)
    board_size = square * 8
    panel_w = max(300, min(420, width - (coord + board_size + 3 * margin)))
    group_w = coord + board_size + margin + panel_w
    left = max(margin, (width - group_w) // 2)
    top = max(margin, (height - board_size - coord) // 2)
    board_rect = pygame.Rect(left + coord, top, board_size, board_size)
    panel_rect = pygame.Rect(board_rect.right + margin, margin, panel_w, height - 2 * margin)
    return Layout((width, height), board_rect, square, panel_rect)


# ----------------------------------------------------------------------------
# Pieces
# ----------------------------------------------------------------------------
class PieceRenderer:
    """Turns piece codes like ``"wn"`` into cached surfaces of any size."""

    def __init__(self, force_vector: Optional[bool] = None) -> None:
        if force_vector is None:
            # Detecting "does this font truly contain playable chess glyphs"
            # is unreliable in practice: many fonts/backends return fallback
            # ("tofu box") metrics for missing glyphs instead of signalling
            # they're missing, so the old auto-detect could pick a font that
            # only draws empty squares for every piece. The hand-drawn vector
            # pieces below work identically on every platform, so they are
            # the default; Unicode glyphs are strictly opt-in.
            force_vector = os.environ.get("CHESS_UNICODE_PIECES") != "1"
        self.font_path: Optional[str] = None if force_vector else self._find_symbol_font()
        self._cache: Dict[Tuple[str, int], pygame.Surface] = {}

    @property
    def mode(self) -> str:
        return "unicode" if self.font_path else "vector"

    @staticmethod
    def _find_symbol_font() -> Optional[str]:
        glyphs = "".join(UNICODE_SOLID.values())
        for name in _SYMBOL_FONT_CANDIDATES:
            try:
                path = pygame.font.match_font(name)
                if not path:
                    continue
                font = pygame.font.Font(path, 32)
                metrics = font.metrics(glyphs)
                if not metrics or not all(m is not None for m in metrics):
                    continue          # font explicitly reports these glyphs as missing

                # Belt-and-braces: some backends report valid-looking metrics
                # for a missing glyph anyway, substituting the *same* "tofu
                # box" placeholder for every character. A king, a pawn and a
                # knight are different shapes/widths in a real chess font, so
                # if every glyph renders to a pixel-identical bounding box
                # it's a placeholder set, not real glyphs -- skip this font.
                bounds = [font.render(g, True, (255, 255, 255)).get_bounding_rect() for g in UNICODE_SOLID.values()]
                sizes = {(b.width, b.height) for b in bounds}
                if len(sizes) > 1:
                    return path
            except Exception:
                continue
        return None

    def get(self, piece: str, size: int) -> pygame.Surface:
        key = (piece, size)
        surf = self._cache.get(key)
        if surf is None:
            surf = self._render_glyph(piece, size) if self.font_path else self._render_vector(piece, size)
            self._cache[key] = surf
        return surf

    # -- unicode glyph rendering ------------------------------------------------
    def _render_glyph(self, piece: str, size: int) -> pygame.Surface:
        white = piece[0] == "w"
        fill = (250, 248, 240) if white else (30, 30, 34)
        edge = (30, 30, 34) if white else (232, 232, 232)
        font = pygame.font.Font(self.font_path, int(size * 0.86))
        glyph = UNICODE_SOLID[piece[1]]
        thickness = max(1, size // 34)
        text = font.render(glyph, True, fill)
        outline = font.render(glyph, True, edge)
        bounds = text.get_bounding_rect()
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        x = (size - bounds.w) // 2 - bounds.x
        y = (size - bounds.h) // 2 - bounds.y
        for dx in (-thickness, 0, thickness):
            for dy in (-thickness, 0, thickness):
                if dx or dy:
                    surf.blit(outline, (x + dx, y + dy))
        surf.blit(text, (x, y))
        return surf

    # -- polygon fallback -----------------------------------------------------------
    def _render_vector(self, piece: str, size: int) -> pygame.Surface:
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        white = piece[0] == "w"
        fill = (250, 248, 240) if white else (44, 44, 50)
        edge = (30, 30, 34) if white else (225, 225, 225)
        lw = max(2, size // 28)
        s = size

        def poly(points):
            pts = [(x * s, y * s) for x, y in points]
            pygame.draw.polygon(surf, fill, pts)
            pygame.draw.polygon(surf, edge, pts, lw)

        def circle(cx, cy, r):
            pygame.draw.circle(surf, fill, (cx * s, cy * s), r * s)
            pygame.draw.circle(surf, edge, (cx * s, cy * s), r * s, lw)

        def thick_line(a, b):
            width = max(4, size // 13)
            pygame.draw.line(surf, edge, (a[0] * s, a[1] * s), (b[0] * s, b[1] * s), width + 2 * lw)
            pygame.draw.line(surf, fill, (a[0] * s, a[1] * s), (b[0] * s, b[1] * s), width)

        def rect(x, y, w, h):
            r = pygame.Rect(x * s, y * s, w * s, h * s)
            pygame.draw.rect(surf, fill, r, border_radius=max(1, size // 40))
            pygame.draw.rect(surf, edge, r, lw, border_radius=max(1, size // 40))

        t = piece[1]
        if t == "p":
            poly([(0.36, 0.76), (0.43, 0.44), (0.57, 0.44), (0.64, 0.76)])
            circle(0.5, 0.31, 0.13)
        elif t == "r":
            # one crenellated silhouette (separate thin rectangles vanish under their outline)
            poly([(0.31, 0.76), (0.34, 0.38), (0.28, 0.38), (0.28, 0.18), (0.39, 0.18),
                  (0.39, 0.27), (0.45, 0.27), (0.45, 0.18), (0.55, 0.18), (0.55, 0.27),
                  (0.61, 0.27), (0.61, 0.18), (0.72, 0.18), (0.72, 0.38), (0.66, 0.38),
                  (0.69, 0.76)])
        elif t == "n":
            poly([(0.30, 0.76), (0.35, 0.52), (0.45, 0.40), (0.36, 0.32), (0.42, 0.20),
                  (0.52, 0.16), (0.57, 0.24), (0.70, 0.34), (0.73, 0.52), (0.67, 0.76)])
            pygame.draw.circle(surf, edge, (0.52 * s, 0.30 * s), max(2, size // 24))
        elif t == "b":
            poly([(0.34, 0.76), (0.42, 0.52), (0.58, 0.52), (0.66, 0.76)])
            poly([(0.5, 0.20), (0.63, 0.40), (0.57, 0.55), (0.43, 0.55), (0.37, 0.40)])
            circle(0.5, 0.16, 0.055)
            pygame.draw.line(surf, edge, (0.5 * s, 0.28 * s), (0.57 * s, 0.38 * s), lw)
        elif t == "q":
            poly([(0.24, 0.76), (0.20, 0.30), (0.35, 0.50), (0.42, 0.24), (0.50, 0.48),
                  (0.58, 0.24), (0.65, 0.50), (0.80, 0.30), (0.76, 0.76)])
            for cx, cy in ((0.20, 0.27), (0.42, 0.20), (0.58, 0.20), (0.80, 0.27)):
                circle(cx, cy, 0.045)
        else:  # king
            poly([(0.28, 0.76), (0.33, 0.42), (0.67, 0.42), (0.72, 0.76)])
            poly([(0.34, 0.42), (0.28, 0.30), (0.72, 0.30), (0.66, 0.42)])
            thick_line((0.5, 0.07), (0.5, 0.30))          # cross: outline first, then fill
            thick_line((0.40, 0.15), (0.60, 0.15))
        rect(0.24, 0.76, 0.52, 0.10)        # common base
        return surf


# ----------------------------------------------------------------------------
# Board
# ----------------------------------------------------------------------------
class BoardRenderer:
    """Draws squares, coordinates, highlights, pieces and the promotion popup."""

    def __init__(self, pieces: PieceRenderer) -> None:
        self.pieces = pieces
        self._overlay_cache: Dict[Tuple[Tuple[int, ...], int], pygame.Surface] = {}
        self._glow_cache: Dict[int, pygame.Surface] = {}

    # -- coordinate mapping -------------------------------------------------------
    @staticmethod
    def square_rect(layout: Layout, square: Square) -> pygame.Rect:
        r, c = square
        s = layout.square
        return pygame.Rect(layout.board_rect.x + c * s, layout.board_rect.y + r * s, s, s)

    @staticmethod
    def square_at(layout: Layout, pos: Tuple[int, int]) -> Optional[Square]:
        rect = layout.board_rect
        if not rect.collidepoint(pos):
            return None
        return (pos[1] - rect.y) // layout.square, (pos[0] - rect.x) // layout.square

    # -- cached overlays ------------------------------------------------------------
    def _overlay(self, rgba: Tuple[int, ...], size: int) -> pygame.Surface:
        key = (rgba, size)
        if key not in self._overlay_cache:
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            surf.fill(rgba)
            self._overlay_cache[key] = surf
        return self._overlay_cache[key]

    def _glow(self, size: int) -> pygame.Surface:
        if size not in self._glow_cache:
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            center = size / 2
            for i in range(size // 2, 0, -1):
                alpha = int(245 * (1 - i / (size / 2)) ** 1.05)
                pygame.draw.circle(surf, (*CHECK_COLOR, alpha), (center, center), i)
            self._glow_cache[size] = surf
        return self._glow_cache[size]

    # -- main draw ---------------------------------------------------------------------
    def draw(self, surface: pygame.Surface, layout: Layout, gs: GameState,
             selected: Optional[Square], targets: Dict[Square, list],
             mouse_pos: Tuple[int, int]) -> None:
        s = layout.square
        board = gs.board
        rect = layout.board_rect

        # board frame / shadow
        frame = rect.inflate(16, 16)
        pygame.draw.rect(surface, (0, 0, 0), frame.move(0, 4), border_radius=10)
        pygame.draw.rect(surface, (66, 52, 42), frame, border_radius=10)

        # squares
        for r in range(8):
            for c in range(8):
                color = LIGHT_SQUARE if (r + c) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(surface, color, (rect.x + c * s, rect.y + r * s, s, s))

        self._draw_coordinates(surface, layout)

        # last move, selection, check
        if gs.last_move is not None:
            for sq in (gs.last_move.from_square, gs.last_move.to_square):
                surface.blit(self._overlay(LAST_MOVE_COLOR, s), self.square_rect(layout, sq))
        if selected is not None:
            surface.blit(self._overlay(SELECTED_COLOR, s), self.square_rect(layout, selected))
        check_sq = gs.king_in_check_square()
        if check_sq is not None:
            glow = self._glow(s)
            surface.blit(glow, self.square_rect(layout, check_sq))

        # pieces
        for r in range(8):
            for c in range(8):
                piece = board.grid[r][c]
                if piece:
                    surface.blit(self.pieces.get(piece, s), self.square_rect(layout, (r, c)))

        # legal move markers (drawn above pieces so captures stay visible)
        hover = self.square_at(layout, mouse_pos)
        for sq in targets:
            sq_rect = self.square_rect(layout, sq)
            marker = pygame.Surface((s, s), pygame.SRCALPHA)
            if board.grid[sq[0]][sq[1]] is not None or any(m.is_en_passant for m in targets[sq]):
                pygame.draw.circle(marker, TARGET_DOT_COLOR, (s // 2, s // 2), int(s * 0.47), max(3, s // 11))
            else:
                pygame.draw.circle(marker, TARGET_DOT_COLOR, (s // 2, s // 2), max(4, int(s * 0.17)))
            if hover == sq:
                marker.fill((255, 255, 255, 60), special_flags=pygame.BLEND_RGBA_ADD)
            surface.blit(marker, sq_rect)

    def _draw_coordinates(self, surface: pygame.Surface, layout: Layout) -> None:
        font = ui_font(max(12, layout.square // 4), bold=True)
        rect, s = layout.board_rect, layout.square
        for i in range(8):
            file_label = font.render("abcdefgh"[i], True, COORD_COLOR)
            surface.blit(file_label, file_label.get_rect(center=(rect.x + i * s + s // 2, rect.bottom + 16)))
            rank_label = font.render(str(8 - i), True, COORD_COLOR)
            surface.blit(rank_label, rank_label.get_rect(center=(rect.x - 16, rect.y + i * s + s // 2)))

    # -- promotion popup -------------------------------------------------------------------
    @staticmethod
    def promotion_choices(layout: Layout, color: str) -> List[Tuple[str, pygame.Rect]]:
        """Rects of the four promotion choices, centred over the board."""
        s = layout.square
        cell = int(s * 1.25)
        gap = s // 6
        total = 4 * cell + 3 * gap
        x0 = layout.board_rect.centerx - total // 2
        y0 = layout.board_rect.centery - cell // 2
        return [(t, pygame.Rect(x0 + i * (cell + gap), y0, cell, cell)) for i, t in enumerate(PROMOTION_TYPES)]

    def draw_promotion_popup(self, surface: pygame.Surface, layout: Layout, color: str,
                             mouse_pos: Tuple[int, int]) -> None:
        dim = pygame.Surface(layout.board_rect.size, pygame.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        surface.blit(dim, layout.board_rect.topleft)
        choices = self.promotion_choices(layout, color)
        first, last = choices[0][1], choices[-1][1]
        card = pygame.Rect(first.left, first.top, last.right - first.left, first.height).inflate(40, 84)
        pygame.draw.rect(surface, (43, 47, 58), card, border_radius=16)
        pygame.draw.rect(surface, (90, 96, 112), card, 2, border_radius=16)
        title = ui_font(max(16, layout.square // 3), bold=True).render("Promote pawn to:", True, (235, 237, 242))
        surface.blit(title, title.get_rect(center=(card.centerx, card.top + 26)))
        for ptype, rect in choices:
            hovered = rect.collidepoint(mouse_pos)
            pygame.draw.rect(surface, (106, 176, 76) if hovered else (62, 68, 84), rect, border_radius=12)
            piece = self.pieces.get(color + ptype, int(rect.width * 0.9))
            surface.blit(piece, piece.get_rect(center=rect.center))
