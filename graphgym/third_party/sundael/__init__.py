# SPDX-FileCopyrightText: 2026 Contributors to the sundael project.
# SPDX-License-Identifier: MPL-2.0
"""Module for disaggregration of consumption and generation."""

import importlib.metadata

from sundael.config import PVConfig
from sundael.disaggregation import PVDisaggregator

# Vendored copy (see VENDOR.md) is imported as a plain subpackage, not a
# pip-installed distribution, so importlib.metadata has no record of it --
# fall back to the version pinned at vendoring time instead of raising
# PackageNotFoundError.
try:
    __version__ = importlib.metadata.version(__package__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "1.0.0+vendored.50e75ab"


__all__ = ["PVConfig", "PVDisaggregator"]
