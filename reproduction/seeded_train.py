#!/usr/bin/env python3
"""Run the repository training entry point with deterministic row sampling."""

from __future__ import annotations

import random
import runpy
from pathlib import Path


random.seed(42)
runpy.run_path(
    str(Path(__file__).resolve().parent.parent / "sim" / "train_ml.py"),
    run_name="__main__",
)
