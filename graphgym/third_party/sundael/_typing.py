# SPDX-FileCopyrightText: 2026 Contributors to the sundael project.
# SPDX-License-Identifier: MPL-2.0
import pandas as pd
import polars as pl

PvReferenceType = pl.DataFrame | pd.DataFrame | pl.Series | pd.Series
