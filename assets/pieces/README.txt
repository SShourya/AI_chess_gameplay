Pieces are drawn programmatically (Unicode glyphs, with a hand-drawn polygon
fallback) rather than loaded from image files — see ui/board_renderer.py's
PieceRenderer. This directory is kept for anyone who wants to drop in their
own PNG/SVG piece set later; it isn't read by the game as shipped.
