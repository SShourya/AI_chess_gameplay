"""Test package.  Putting the project root on sys.path here lets the tests run
with ``python -m unittest`` or ``pytest`` from the project root."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
