# SPDX-FileCopyrightText: 2026 Contributors to the sundael project.
# SPDX-License-Identifier: MPL-2.0
"""Obtain temperature from KNMI API or CSV files."""

import json
from collections.abc import Iterable
from functools import singledispatch
from importlib.resources import as_file, files
from pathlib import Path
from typing import Literal

import polars as pl
import requests
import structlog
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sundael.cache import cache

logger = structlog.get_logger(__name__)


def _get_knmi_hourly_temperature_series(
    start_date: str, end_date: str, station_ids: Iterable[str | int] = (260,), fmt: Literal["json", "csv"] = "json"
) -> str:
    """Retrieve hourly temperatures from KNMI API.

    Args:
        start_date (str): Start of range in YYYYMMDD format.
        end_date (str): End of range in YYYYMMDD format.
        station_ids (Iterable, optional): One or more station identifiers.
            The default is 260 (de Bilt).
        fmt (str): Return format, either "json" or "csv".

    Returns:
        String with temperature data in the specified format.
    """
    start = start_date + "00"
    end = end_date + "00"
    url = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"
    params = {"stns": ":".join(map(str, station_ids)), "vars": "TEMP", "start": start, "end": end, "fmt": fmt}
    logger.info(f"Accessing KNMI API with period {start} - {end}")
    retry_strategy = Retry(
        total=5,
        allowed_methods=["POST"],
        backoff_factor=10,
        backoff_jitter=5,
        status_forcelist=[408, 429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    response = session.post(url, json=params)
    if not response.ok:
        msg = f"Accessing KNMI API failed: {response}"
        raise OSError(msg)

    return response.text


get_knmi_hourly_temperature_series = cache.cache(_get_knmi_hourly_temperature_series)


def read_knmi_csv(fname: Path | str) -> pl.DataFrame:
    """Read CSV file in KNMI format.

    Args:
        fname (str or Path): Filename of CSV file, formatted in KNMI format.

    Returns:
        DataFrame with three columns: `station_id`, `datetime` and `temperature
            (in degrees Celsius).
    """
    # bearer:disable
    with open(fname) as f:
        skip_lines = len([line for line in f.readlines() if line.startswith("#")]) - 1

    df = (
        pl.read_csv(fname, skip_lines=skip_lines)
        .rename({"YYYYMMDD": "date", "   HH": "hour", "# STN": "station_code", "    T": "T"})
        .with_columns(
            pl.col("date").cast(pl.String),
            pl.col("station_code").str.strip_chars(),
            pl.col("hour").str.strip_chars().cast(pl.Int32),
            pl.col("T").str.strip_chars().cast(pl.Int32),
        )
    )
    return df.pipe(format_knmi)


def format_knmi(df: pl.DataFrame) -> pl.DataFrame:
    """Format columns based on KNMI-specific input.

    Adds a datetime column and a temperature column (degrees Celsius).

    Args:
        df (DataFrame): DataFrame with `date`, `hour` and `T` columns.

    Returns:
        DataFrame with added `datetime` and `temperature` columns.
    """
    return (
        df.with_columns(
            pl.col("date")
            .str.slice(0, 10)
            .add(" ")
            .add(pl.col("hour").sub(1).cast(pl.String).str.pad_start(2, "0").add(":00"))
            .str.replace_all("-", "")
            .str.to_datetime(time_zone=None, format="%Y%m%d %H:%M")
            .dt.replace_time_zone("Europe/Amsterdam", non_existent="null", ambiguous="null")
            .dt.convert_time_zone("UTC")
            .alias("datetime")
        )
        .with_columns(
            pl.col("T").truediv(10).alias("temperature"),
        )
        .select("station_code", "datetime", "temperature")
    )


@singledispatch
def load_temperature_data(source: int | str | Path) -> pl.DataFrame:  # pragma: nocover
    """Load yearly temperature data from KNMI API or CSV file.

    Args:
        source (int, str or Path): If source is specified as `2023` it will load
            the temperature data for that year from the KNMI API. If source is a
            strint, it will parse it as a filename.

    Returns:
        DataFrame with three columns: `station_id`, `datetime` and `temperature
            (in degrees Celsius).
    """
    msg = "Invalid data type for source"
    raise NotImplementedError(msg)


@load_temperature_data.register
def _(source: int) -> pl.DataFrame:
    """Load yearly temperature data from KNMI API."""
    data = json.loads(get_knmi_hourly_temperature_series(f"{source}0101", f"{source}1231"))
    return pl.DataFrame(data).pipe(format_knmi)


@load_temperature_data.register
def _(source: str | Path) -> pl.DataFrame:
    """Load yearly temperature data from KNMI CSV file."""
    if Path(source).exists():
        return read_knmi_csv(source)

    # bearer:disable
    with as_file(files("sundael.data").joinpath(str(source))) as packaged_file:
        if not packaged_file.exists():
            msg = f"could not find file {source} locally or in package data"
            raise FileNotFoundError(msg)

        result = read_knmi_csv(packaged_file)

    return result
