"""Translating raw Pygame events into game actions.

The handler owns the *interaction state* (which piece is selected, which
squares it can go to, whether a promotion choice is pending) but never mutates
the board itself: every move goes through ``GameState.play``, which re-checks
legality.  The human is locked out whenever it is not their turn or the
computer is thinking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import pygame

from game.board import Square
from game.game_state import GameState
from game.move import Move
from ui.board_renderer import BoardRenderer, Layout
from ui.menu import GameOverDialog, SidePanel


@dataclass
class Interaction:
    """Transient UI state for the human player."""
    selected: Optional[Square] = None
    targets: Dict[Square, List[Move]] = field(default_factory=dict)
    pending_promotion: Optional[Tuple[Square, Square]] = None

    def clear(self) -> None:
        self.selected = None
        self.targets = {}
        self.pending_promotion = None


class InputHandler:
    def __init__(
        self,
        gs: GameState,
        renderer: BoardRenderer,
        panel: SidePanel,
        dialog: GameOverDialog,
        interaction: Interaction,
        is_ai_busy: Callable[[], bool],
        on_restart: Callable[[], None],
        on_quit: Callable[[], None],
    ) -> None:
        self.gs = gs
        self.renderer = renderer
        self.panel = panel
        self.dialog = dialog
        self.interaction = interaction
        self.is_ai_busy = is_ai_busy
        self.on_restart = on_restart
        self.on_quit = on_quit

    # ---------------------------------------------------------------- events
    def handle_event(self, event: pygame.event.Event, layout: Layout) -> None:
        if event.type == pygame.QUIT:
            self.on_quit()
        elif event.type == pygame.KEYDOWN:
            self._on_key(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._on_click(event.pos, layout)

    def _on_key(self, key: int) -> None:
        if key in (pygame.K_r, pygame.K_n):
            self.on_restart()                      # R = restart, N = new game
        elif key == pygame.K_q:
            self.on_quit()
        elif key == pygame.K_ESCAPE:
            self.interaction.clear()               # cancel selection / promotion popup

    def _on_click(self, pos: Tuple[int, int], layout: Layout) -> None:
        gs, ui = self.gs, self.interaction
        if gs.game_over:
            self.dialog.handle_click(pos)
            return
        if self.panel.handle_click(pos):
            return
        if ui.pending_promotion is not None:
            self._click_promotion(pos, layout)
            return
        if not gs.is_human_turn or self.is_ai_busy():
            return                                  # locked out while the computer thinks
        self._click_board(pos, layout)

    # ---------------------------------------------------------------- board clicks
    def _click_board(self, pos: Tuple[int, int], layout: Layout) -> None:
        gs, ui = self.gs, self.interaction
        square = self.renderer.square_at(layout, pos)
        if square is None:
            ui.clear()
            return

        if ui.selected is not None and square in ui.targets:
            origin = ui.selected
            if gs.needs_promotion_choice(origin, square):
                ui.pending_promotion = (origin, square)      # ask which piece
            else:
                gs.play(origin, square)
                ui.clear()
            return

        piece = gs.board.piece_at(square)
        if piece is not None and piece[0] == gs.human_color:
            ui.selected = square
            ui.targets = {}
            for move in gs.legal_moves_from(square):
                ui.targets.setdefault(move.to_square, []).append(move)
        else:
            ui.clear()

    def _click_promotion(self, pos: Tuple[int, int], layout: Layout) -> None:
        gs, ui = self.gs, self.interaction
        for ptype, rect in self.renderer.promotion_choices(layout, gs.human_color):
            if rect.collidepoint(pos):
                origin, target = ui.pending_promotion
                gs.play(origin, target, ptype)
                ui.clear()
                return
        ui.clear()                                  # clicked elsewhere: cancel
