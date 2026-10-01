# SPDX-FileCopyrightText: 2026 Contributors to the sundael project.
# SPDX-License-Identifier: MPL-2.0
"""Disk-based caching."""

from pathlib import Path
from tempfile import gettempdir

from joblib import Memory

cache = Memory(location=Path(gettempdir()) / "sundael")
