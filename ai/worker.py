"""Run the AI search on a background thread so the Pygame window stays
responsive while the computer is thinking.

Design
------
* The worker receives a **copy** of the board, so the UI thread can keep
  drawing the real board while the search makes and unmakes thousands of moves
  on its private copy.  No locks are needed for the board itself.
* Each request is a ``_Job`` with its own cancel ``threading.Event``.  If the
  player restarts the game mid-search the job is cancelled, and even if the old
  thread takes a moment to notice, its result is discarded because it is no
  longer the *current* job.
* The UI polls :meth:`AIWorker.poll` once per frame; it never blocks.
"""

from __future__ import annotations

import threading
import traceback
from typing import Optional

from ai.minimax import SearchCancelled, SearchResult, find_best_move
from game.board import Board


class _Job:
    def __init__(self) -> None:
        self.cancel_event = threading.Event()
        self.result: Optional[SearchResult] = None
        self.error: Optional[str] = None
        self.finished = threading.Event()


class AIWorker:
    """Owns at most one running search at a time."""

    def __init__(self) -> None:
        self._job: Optional[_Job] = None

    @property
    def busy(self) -> bool:
        """True while a search is running or its result is waiting to be collected."""
        return self._job is not None

    def start(self, board: Board, depth: int, ai_color: str, randomize: bool = True) -> None:
        """Begin searching a private copy of ``board``."""
        self.cancel()
        job = _Job()
        self._job = job
        snapshot = board.copy()

        def run() -> None:
            try:
                job.result = find_best_move(
                    snapshot, depth, ai_color, randomize=randomize,
                    cancel_event=job.cancel_event,
                )
            except SearchCancelled:
                job.result = None
            except Exception:                       # never let a bug kill the thread silently
                job.error = traceback.format_exc()
            finally:
                job.finished.set()

        threading.Thread(target=run, name="chess-ai", daemon=True).start()

    def poll(self) -> Optional[SearchResult]:
        """Return the finished result (once), or ``None`` if still thinking.

        Raises ``RuntimeError`` if the search thread crashed."""
        job = self._job
        if job is None or not job.finished.is_set():
            return None
        self._job = None
        if job.error:
            raise RuntimeError(f"AI search failed:\n{job.error}")
        return job.result

    def cancel(self) -> None:
        """Abort the current search (if any) and forget about it."""
        if self._job is not None:
            self._job.cancel_event.set()
            self._job = None
