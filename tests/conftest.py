"""
tests/conftest.py
-----------------
Shared pytest fixtures for the Tox21 QSAR test suite.
"""
import sys
from pathlib import Path

# Ensure project root is on the Python path so `src.*` imports resolve correctly
sys.path.insert(0, str(Path(__file__).parent.parent))
