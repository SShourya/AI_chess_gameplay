"""Menu-style widgets: buttons, the difficulty selector, the information side
panel (turn, status, captured pieces, AI statistics, move list) and the
game-over dialog."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import pygame

from ai.minimax import DIFFICULTY_DEPTHS, SearchStats, format_score
from game.game_state import GameState, GameStatus
from game.pieces import PIECE_VALUES, BLACK, WHITE, color_name
from ui.board_renderer import Layout, PieceRenderer, ui_font

TEXT = (235, 237, 242)
TEXT_DIM = (150, 156, 170)
CARD = (43, 47, 58)
CARD_BORDER = (64, 70, 86)
GREEN = (106, 176, 76)
GREEN_HOVER = (128, 198, 96)
GRAY = (74, 80, 98)
GRAY_HOVER = (94, 101, 122)
RED = (205, 80, 80)
RED_HOVER = (225, 100, 100)
AMBER = (240, 190, 80)


@dataclass
class AIInfo:
    """What the panel needs to know about the computer player."""
    thinking: bool = False
    stats: Optional[SearchStats] = None
    score: float = 0.0
    move_text: str = ""
    thinking_since: float = 0.0


class Button:
    """A rounded rectangular button with hover feedback."""

    STYLES = {
        "primary": (GREEN, GREEN_HOVER, (20, 30, 15)),
        "secondary": (GRAY, GRAY_HOVER, TEXT),
        "danger": (RED, RED_HOVER, (255, 255, 255)),
    }

    def __init__(self, label: str, on_click: Callable[[], None], style: str = "secondary") -> None:
        self.label = label
        self.on_click = on_click
        self.style = style
        self.rect = pygame.Rect(0, 0, 0, 0)

    def draw(self, surface: pygame.Surface, mouse_pos: Tuple[int, int], size: int = 18) -> None:
        base, hover, text_color = self.STYLES[self.style]
        hovered = self.rect.collidepoint(mouse_pos)
        rect = self.rect.inflate(4, 4) if hovered else self.rect
        pygame.draw.rect(surface, hover if hovered else base, rect, border_radius=10)
        label = ui_font(size, bold=True).render(self.label, True, text_color)
        surface.blit(label, label.get_rect(center=rect.center))

    def click(self, pos: Tuple[int, int]) -> bool:
        if self.rect.collidepoint(pos):
            self.on_click()
            return True
        return False


class SegmentedControl:
    """A row of mutually exclusive options (used for the difficulty)."""

    def __init__(self, options: List[str], selected: str, on_change: Callable[[str], None]) -> None:
        self.options = options
        self.selected = selected
        self.on_change = on_change
        self.rect = pygame.Rect(0, 0, 0, 0)

    def _segments(self) -> List[Tuple[str, pygame.Rect]]:
        w = self.rect.width // len(self.options)
        return [(o, pygame.Rect(self.rect.x + i * w, self.rect.y, w, self.rect.height))
                for i, o in enumerate(self.options)]

    def draw(self, surface: pygame.Surface, mouse_pos: Tuple[int, int]) -> None:
        pygame.draw.rect(surface, (34, 37, 46), self.rect, border_radius=10)
        for name, seg in self._segments():
            active = name == self.selected
            hovered = seg.collidepoint(mouse_pos)
            if active:
                pygame.draw.rect(surface, GREEN, seg.inflate(-6, -6), border_radius=8)
            elif hovered:
                pygame.draw.rect(surface, GRAY, seg.inflate(-6, -6), border_radius=8)
            color = (20, 30, 15) if active else TEXT
            label = ui_font(16, bold=True).render(name, True, color)
            depth = ui_font(12).render(f"depth {DIFFICULTY_DEPTHS[name]}", True, color if active else TEXT_DIM)
            surface.blit(label, label.get_rect(center=(seg.centerx, seg.centery - 7)))
            surface.blit(depth, depth.get_rect(center=(seg.centerx, seg.centery + 10)))

    def click(self, pos: Tuple[int, int]) -> bool:
        for name, seg in self._segments():
            if seg.collidepoint(pos):
                if name != self.selected:
                    self.selected = name
                    self.on_change(name)
                return True
        return False


class SidePanel:
    """Everything to the right of the board."""

    def __init__(self, pieces: PieceRenderer, difficulty: str,
                 on_restart: Callable[[], None], on_quit: Callable[[], None],
                 on_difficulty: Callable[[str], None]) -> None:
        self.pieces = pieces
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.difficulty = SegmentedControl(list(DIFFICULTY_DEPTHS), difficulty, on_difficulty)
        self.restart_button = Button("Restart Game", on_restart, "primary")
        self.quit_button = Button("Quit", on_quit, "danger")

    # ---------------------------------------------------------------- input
    def handle_click(self, pos: Tuple[int, int]) -> bool:
        if not self.rect.collidepoint(pos):
            return False
        self.difficulty.click(pos)
        self.restart_button.click(pos)
        self.quit_button.click(pos)
        return True

    # ---------------------------------------------------------------- drawing
    def draw(self, surface: pygame.Surface, panel_rect: pygame.Rect, gs: GameState,
             ai: AIInfo, mouse_pos: Tuple[int, int]) -> None:
        self.rect = panel_rect
        x, w = panel_rect.x, panel_rect.width
        pad = 14
        gap = 10
        y = panel_rect.y

        # ---- bottom-anchored buttons ------------------------------------------------
        btn_h = 40
        self.quit_button.rect = pygame.Rect(x, panel_rect.bottom - btn_h, w, btn_h)
        self.restart_button.rect = pygame.Rect(x, self.quit_button.rect.y - btn_h - gap, w, btn_h)
        bottom_limit = self.restart_button.rect.y - gap

        # ---- title ------------------------------------------------------------------
        title = ui_font(28, bold=True).render("Chess AI", True, TEXT)
        surface.blit(title, (x, y - 2))
        sub = ui_font(13).render("Minimax  +  Alpha-Beta pruning", True, TEXT_DIM)
        surface.blit(sub, (x + 2, y + 32))
        y += 58

        # Short windows: shrink the cards so the AI-statistics card (a required
        # element) always fits; the optional move list is dropped first instead.
        compact = (bottom_limit - y) < 470
        turn_h, captured_h, stats_h = (74, 108, 118) if compact else (82, 122, 128)

        # ---- turn / status card -----------------------------------------------------
        y = self._draw_turn_card(surface, gs, ai, x, y, w, pad, turn_h) + gap

        # ---- captured pieces -----------------------------------------------------------
        y = self._draw_captured_card(surface, gs, x, y, w, pad, captured_h) + gap

        # ---- difficulty ---------------------------------------------------------------------
        card = pygame.Rect(x, y, w, 88)
        self._card(surface, card)
        surface.blit(ui_font(14, bold=True).render("Difficulty", True, TEXT_DIM), (x + pad, y + 8))
        self.difficulty.rect = pygame.Rect(x + pad, y + 32, w - 2 * pad, 46)
        self.difficulty.draw(surface, mouse_pos)
        y += card.height + gap

        # ---- AI statistics -------------------------------------------------------------------
        self._draw_stats_card(surface, ai, pygame.Rect(x, y, w, stats_h), pad, compact)
        y += stats_h + gap

        # ---- recent moves (only if there is room left) -----------------------------------------
        if bottom_limit - y >= 70:
            self._draw_moves_card(surface, gs, pygame.Rect(x, y, w, bottom_limit - y), pad)

        self.restart_button.draw(surface, mouse_pos)
        self.quit_button.draw(surface, mouse_pos)

    # ---- sections -----------------------------------------------------------------------------
    @staticmethod
    def _card(surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, CARD, rect, border_radius=12)
        pygame.draw.rect(surface, CARD_BORDER, rect, 1, border_radius=12)

    def _draw_turn_card(self, surface, gs: GameState, ai: AIInfo, x, y, w, pad, height=82) -> int:
        card = pygame.Rect(x, y, w, height)
        self._card(surface, card)
        side = gs.turn
        dot_color = (245, 245, 240) if side == WHITE else (25, 25, 28)
        dot_y = card.y + height // 2 - 1
        pygame.draw.circle(surface, (120, 126, 140), (card.x + 30, dot_y), 15)
        pygame.draw.circle(surface, dot_color, (card.x + 30, dot_y), 13)

        if gs.game_over:
            headline = "Game over"
        elif gs.is_human_turn:
            headline = f"Turn: {color_name(side)}  (you)"
        else:
            headline = f"Turn: {color_name(side)}  (computer)"
        surface.blit(ui_font(20, bold=True).render(headline, True, TEXT), (card.x + 56, card.y + height // 2 - 27))

        if ai.thinking and gs.is_ai_turn:
            dots = "." * (int((time.time() - ai.thinking_since) * 3) % 4)
            prefix = "Check! " if gs.status == GameStatus.CHECK else ""
            status, color = f"{prefix}Computer is thinking{dots}", AMBER
        else:
            status = gs.status_text()
            color = RED if gs.status in (GameStatus.CHECK, GameStatus.CHECKMATE) else \
                AMBER if gs.status in (GameStatus.STALEMATE, GameStatus.DRAW) else TEXT_DIM
        surface.blit(ui_font(16).render(status, True, color), (card.x + 56, card.y + height // 2 + 5))
        return card.bottom

    def _draw_captured_card(self, surface, gs: GameState, x, y, w, pad, height=122) -> int:
        card = pygame.Rect(x, y, w, height)
        row_gap = 52 if height >= 120 else 46
        self._card(surface, card)
        balance = gs.material_balance()
        rows = ((WHITE, "White captured", balance > 0, balance),
                (BLACK, "Black captured", balance < 0, -balance))
        row_y = card.y + 10
        for side, label, leading, lead in rows:
            surface.blit(ui_font(14, bold=True).render(label, True, TEXT_DIM), (card.x + pad, row_y))
            if leading:
                tag = ui_font(14, bold=True).render(f"+{lead}", True, GREEN)
                surface.blit(tag, (card.right - pad - tag.get_width(), row_y))
            pieces = sorted(gs.captured[side], key=lambda p: -PIECE_VALUES[p[1]])
            step = 20
            for i, piece in enumerate(pieces):
                icon = self.pieces.get(piece, 30)
                px = card.x + pad + (i * step) % (w - 2 * pad - 30)
                surface.blit(icon, (px, row_y + 20))
            row_y += row_gap
        return card.bottom

    def _draw_stats_card(self, surface, ai: AIInfo, card: pygame.Rect, pad, compact: bool = False) -> None:
        self._card(surface, card)
        surface.blit(ui_font(14, bold=True).render("AI Statistics", True, TEXT_DIM), (card.x + pad, card.y + 8))
        s = ai.stats
        if s is None:
            text = ui_font(15).render("No computer move yet.", True, TEXT_DIM)
            surface.blit(text, (card.x + pad, card.y + 36))
            return
        lines = [
            ("Depth", f"{s.depth}"),
            ("Nodes searched", f"{s.nodes:,}"),
            ("Nodes pruned", f"{s.pruned:,}"),
            ("Evaluation", format_score(ai.score)),
            ("Time", f"{s.elapsed:.2f} sec"),
        ]
        for i, (label, value) in enumerate(lines):
            ly = card.y + 30 + i * (17 if compact else 19)
            surface.blit(ui_font(15).render(label, True, TEXT_DIM), (card.x + pad, ly))
            val = ui_font(15, bold=True).render(value, True, TEXT)
            surface.blit(val, (card.right - pad - val.get_width(), ly))

    def _draw_moves_card(self, surface, gs: GameState, card: pygame.Rect, pad) -> None:
        self._card(surface, card)
        surface.blit(ui_font(14, bold=True).render("Recent moves", True, TEXT_DIM), (card.x + pad, card.y + 8))
        log = gs.move_log
        first_ply = max(0, len(log) - 2 * max(1, (card.height - 40) // 18))
        first_ply -= first_ply % 2
        row = 0
        for ply in range(first_ply, len(log), 2):
            number = ply // 2 + 1
            white = log[ply]
            black = log[ply + 1] if ply + 1 < len(log) else ""
            text = ui_font(15).render(f"{number:>3}.  {white:<7} {black}", True, TEXT)
            surface.blit(text, (card.x + pad, card.y + 30 + row * 18))
            row += 1


class GameOverDialog:
    """Modal card shown over the board when the game has ended."""

    def __init__(self, on_play_again: Callable[[], None], on_quit: Callable[[], None]) -> None:
        self.play_again = Button("Play Again", on_play_again, "primary")
        self.quit = Button("Quit", on_quit, "secondary")
        self.card = pygame.Rect(0, 0, 0, 0)

    def handle_click(self, pos: Tuple[int, int]) -> bool:
        return self.play_again.click(pos) or self.quit.click(pos)

    def draw(self, surface: pygame.Surface, layout: Layout, gs: GameState, mouse_pos: Tuple[int, int]) -> None:
        board = layout.board_rect
        dim = pygame.Surface(board.size, pygame.SRCALPHA)
        dim.fill((0, 0, 0, 165))
        surface.blit(dim, board.topleft)

        card_w = min(board.width - 30, 420)
        self.card = pygame.Rect(0, 0, card_w, 250)
        self.card.center = board.center
        pygame.draw.rect(surface, CARD, self.card, border_radius=18)
        pygame.draw.rect(surface, CARD_BORDER, self.card, 2, border_radius=18)

        if gs.status == GameStatus.CHECKMATE:
            title, message, color = "CHECKMATE!", f"{color_name(gs.winner)} wins.", GREEN if gs.winner == gs.human_color else RED
        elif gs.status == GameStatus.STALEMATE:
            title, message, color = "STALEMATE", "The game is a draw.", AMBER
        else:
            title, message, color = "DRAW", gs.draw_reason + ".", AMBER

        t = ui_font(40, bold=True).render(title, True, color)
        surface.blit(t, t.get_rect(center=(self.card.centerx, self.card.y + 52)))
        m = ui_font(22).render(message, True, TEXT)
        surface.blit(m, m.get_rect(center=(self.card.centerx, self.card.y + 100)))
        if gs.status == GameStatus.CHECKMATE:
            hint = "You win - well played!" if gs.winner == gs.human_color else "The computer wins this one."
            h = ui_font(15).render(hint, True, TEXT_DIM)
            surface.blit(h, h.get_rect(center=(self.card.centerx, self.card.y + 128)))

        bw = (self.card.width - 60) // 2
        self.play_again.rect = pygame.Rect(self.card.x + 20, self.card.bottom - 66, bw, 44)
        self.quit.rect = pygame.Rect(self.play_again.rect.right + 20, self.card.bottom - 66, bw, 44)
        self.play_again.draw(surface, mouse_pos, 20)
        self.quit.draw(surface, mouse_pos, 20)
