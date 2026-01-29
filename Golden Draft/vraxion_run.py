#!/usr/bin/env python
"""VRAXION run entrypoint (Golden Draft).

This is the modern replacement for the legacy monolithic training script.

It intentionally lives in Golden Draft (not the end-user "DVD" package) because
it wires together tooling, datasets, and training orchestration.
"""

from __future__ import annotations

from tools.instnct_runner import main


if __name__ == "__main__":
    main()

