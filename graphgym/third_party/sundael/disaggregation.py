# SPDX-FileCopyrightText: 2026 Contributors to the sundael project.
# SPDX-License-Identifier: MPL-2.0
"""Disaggregation functions to estimate generation and consumption from import and export.

Import is designated as net consumption (net_con) and export als net generation (net_gen).
"""

import json
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import cached_property
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import polars.selectors as cs
import structlog
from numba import njit, prange

from sundael._typing import PvReferenceType
from sundael.config import (
    DEFAULT_CONFIG,
    PVConfig,
)
from sundael.solar import SolarProperties, estimate_parameters, solar_maxgen_parallel
from sundael.temperature import format_knmi, get_knmi_hourly_temperature_series, load_temperature_data
from sundael.utils import mean_gt_zero

logger = structlog.get_logger(__name__)

# Number of measurements to use when subsampling
N_SAMPLE = 6587
MP_CONTEXT = multiprocessing.get_context("spawn")


class PVDisaggregator:
    """Disaggregate generation and consumption.

    Generation is determined based on or more average PV profile(s). If available,
    average PV profiles can be supplied by area, for example per postal code.
    The profiles are corrected for installation size, tilt and orientation. These
    are calculated based on the SunDance algorithm.

    Args:
        pv_reference (DataFrame or Series): Reference ratio (indexed by PC4).
        pv_ratio (pd.DataFrame): Reference ratio (indexed by PC4).

    Examples:
        Read the data files.

        >>> pv_ratio = pd.read_parquet("pv_ratio_pc4.parquet").T
        >>> net_gen = pd.read_parquet("E1_net_gen_wide_1.parquet").T
        >>> net_con = pd.read_parquet("E1_net_con_wide_1.parquet").T

        Run the disaggregation.

        >>> pv = PVDisaggregator(pv_ratio)
        >>> result = pv.disaggregate(net_con, net_gen)
    """

    _logged_knmi_warning = False

    def __init__(
        self,
        *,
        pv_reference: PvReferenceType | None = None,
        pv_ratio: PvReferenceType | None = None,
        area_mapping: dict[str, str] | None = None,
        area_latlon: pd.DataFrame | None = None,
        temperature: pd.Series | pl.Series | None = None,
        config: PVConfig | None = None,
    ) -> None:
        """Disaggregate generation and consumption."""
        self.config = config if config is not None else DEFAULT_CONFIG
        self.time_stamps = pd.DatetimeIndex([])
        self._set_reference(pv_reference, pv_ratio)
        self.area_mapping = area_mapping
        self.area_latlon = area_latlon
        self._area_solar_values: dict[str, SolarProperties] = {}
        self.temperature = temperature
        self._mpi = pd.DataFrame()

    def _set_reference(
        self,
        pv_reference: PvReferenceType | None = None,
        pv_ratio: PvReferenceType | None = None,
    ) -> None:
        """Set the PV ratio, either directly or by using a reference."""
        if pv_reference is None and pv_ratio is None:
            # Don't set ratio
            return

        if pv_reference is not None and pv_ratio is not None:
            msg = "Please supply either a reference or a ratio"
            raise ValueError(msg)

        self.pv_ratio = (
            PVDisaggregator.convert_pv_reference_to_ratio(cast(PvReferenceType, pv_reference))
            if pv_ratio is None
            else pv_ratio
        )

    @property
    def pv_ratio(self) -> pd.DataFrame:
        """DataFrame: PV ratio."""
        # If a ratio was never supplied, default to a ratio of 1.0 for every time stamp.
        # Build it on the fly (without caching) so it always reflects the current
        # time stamps, which may only become known once disaggregate() is called.
        if not hasattr(self, "_pv_ratio"):
            return pd.DataFrame(1.0, columns=self.time_stamps, index=["max"])

        return self._pv_ratio

    @pv_ratio.setter
    def pv_ratio(self, value: PvReferenceType) -> None:
        """Setter for pv_ratio."""
        if isinstance(value, pl.DataFrame):
            dt_columns = value.select(cs.datetime()).columns
            if len(dt_columns) == 0:
                msg = "PV ratio needs a DateTime column!"
                raise ValueError(msg)
            if len(dt_columns) > 1:
                msg = "PV ratio needs exactly one DateTime column!"
                raise ValueError(msg)

            value = value.to_pandas().set_index(dt_columns[0])
        if isinstance(value, pd.Series):
            value = value.to_frame()

        if isinstance(value.index, pd.DatetimeIndex):
            value = value.T

        if not isinstance(value.columns, pd.DatetimeIndex):
            msg = "DataFrame needs a DateTime index!"
            raise ValueError(msg)

        self._pv_ratio = value
        self.time_stamps = self._pv_ratio.columns
        # Ensure the DataFrame uses the properly localized timestamps
        self._pv_ratio.columns = self.time_stamps

    @property
    def area_mapping(self) -> dict[str, str] | None:
        """dict: area mapping."""
        return self._area_mapping

    @area_mapping.setter
    def area_mapping(self, value: dict[str, str] | None) -> None:
        """Setter for area_mapping."""
        if value is None:
            if self.pv_ratio.shape[0] > 1:
                msg = "Please provide an area mapping if pv_ratio contains more than one area!"
                raise ValueError(msg)
            elif self.pv_ratio.shape[0] == 1:
                default_value = self.pv_ratio.index[0]
                value = defaultdict(lambda: default_value)
            else:
                value = None  # Mostly for testing
        self._area_mapping = value

    @cached_property
    def _default_solar_values(self) -> SolarProperties:
        return SolarProperties(
            time_stamps=self.time_stamps,
            temperature=self.temperature,
            config=self.config,
        )

    @property
    def years(self) -> list[int]:
        """years: list of unique years in the time_stamps."""
        return list(set(self.time_stamps.year))

    @classmethod
    def convert_pv_reference_to_ratio(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Convert PV reference to ratio.

        Args:
            df (DataFrame): DataFrame with reference PV values. One of the indices,
                index or column, needs to be a DatetimeIndex. The other index should
                contain unique identifiers, for instance a postal code. The most simple
                input DataFrame contains only one identifier, which is an average PV profile
                for a specific year.

        Returns:
            DataFrame: Reference PV ratios, used in the disaggregation algorithm.
                The DataFrame is in wide format, with the columns as DatetimeIndex.
        """
        logger.info("Calculating pv ratio from pv reference")

        if isinstance(df.index, pd.DatetimeIndex):
            net_gen = df.T.copy()
        elif isinstance(df.columns, pd.DatetimeIndex):
            net_gen = df.copy()
        else:
            msg = "Either the index or the columns need to be a DatetimeIndex"
            raise ValueError(msg)

        # Fake ratio, all ones
        max_ratio = net_gen.copy()
        max_ratio.loc[:, :] = 1

        net_gen = net_gen.div(net_gen.max(1), axis=0)
        # net_gen.index = net_gen.index + "|" + [str(x) for x in range(1, net_gen.shape[0] + 1)]

        # Fake net_con, all zeros
        net_con = net_gen.copy()
        net_con.loc[:, :] = 0.0

        # Run the disaggregation
        pv = PVDisaggregator(
            pv_ratio=max_ratio.T,
            area_mapping={k: k for k in max_ratio.index},
        )
        result = pv.disaggregate(net_con, net_gen, include_self_consumption=False)
        generation = result["generation"]

        # Convert results
        ratio = net_gen / generation
        # ratio.index = ratio.index.to_series().str.split("|", expand=True).iloc[:, 0]
        ratio[ratio > 1] = 1
        ratio[ratio < 0] = 0
        ratio = ratio.fillna(0)

        return ratio

    @property
    def area_latlon(self) -> pd.DataFrame:
        """DataFrame with latitude and longitude per postal code."""
        return self._area_latlon

    @area_latlon.setter
    def area_latlon(self, df: pd.DataFrame | None) -> None:
        if df is not None:
            df.index = df.index.astype(str)
            df = df.rename({"lat": "latitude", "lon": "longitude"})
        self._area_latlon = df

    @property
    def mpi(self) -> pd.DataFrame:
        """DataFrame: performance indicators."""
        # Remove PC4.
        # In the end, the customer/pc4 mapping should be handled in a better way
        # and then this can be removed.
        mpi = self._mpi.reset_index()
        mpi["variable"] = mpi["variable"].astype(str).str.replace(r"^.+\|", "", regex=True)
        return mpi

    @property
    def time_stamps(self) -> pd.DatetimeIndex:
        """DateTimeIndex of time stamps."""
        return self._time_stamps

    @time_stamps.setter
    def time_stamps(self, value: pd.DatetimeIndex) -> None:
        self._time_stamps = value
        if self._time_stamps.tz is None:
            self._time_stamps = self._time_stamps.tz_localize("UTC")
        else:
            self._time_stamps = self._time_stamps.tz_convert("UTC")

        # Update temperature if it was not correctly set yet
        if len(self.temperature) == 0:
            self.temperature = None

    @property
    def temperature(self) -> npt.NDArray[np.float64]:
        """array: temperature in degrees Celsius."""
        return self._temperature if hasattr(self, "_temperature") else np.array([])

    @temperature.setter
    def temperature(self, value: npt.NDArray[np.float64] | pl.Series | pd.Series | float | None) -> None:
        """Setter for temperature property."""
        match value:
            case None:
                value = (
                    self._set_temperature()
                    if self.config.auto_infer_temp
                    else np.full(self.time_stamps.shape, self.config.yearly_avg_temp)
                )
            case float():
                value = np.full(self.time_stamps.shape, value)
            case pd.Series():
                value = cast(npt.NDArray[np.float64], value.values)
            case pl.Series():
                value = cast(npt.NDArray[np.float64], value.to_numpy())
            case np.ndarray():
                value = cast(npt.NDArray[np.float64], value)
            case _:
                msg = "Unknown type for temperature"
                raise ValueError(msg)

        if value.shape != self.time_stamps.shape:
            logger.error("Wrong shape for temperature!")
            logger.error(f"Expected shape {self.time_stamps.shape} based on datetime range, got shape {value.shape}.")
            msg = "Wrong shape for temperature!"
            raise ValueError(msg)

        self._temperature = value

    def _set_temperature(self) -> npt.NDArray[np.float64]:
        """Set temperature based on file or KNMI data."""
        if not self._logged_knmi_warning:
            logger.info("Setting temperature based on data from the Dutch meteorological institute.")
            logger.info("This is not valid for other localities! Set auto_infer_temp to `False` ")
            logger.info("to disable or provide temperature to PVDisaggregator.")
            self._logged_knmi_warning = True

        if len(self.years) == 1:
            try:
                df = load_temperature_data(f"{self.years[0]}.KNMI.temperature.csv")
            except FileNotFoundError:
                df = load_temperature_data(self.years[0])
        elif len(self.years) == 0:
            return np.array([])
        else:
            df = pl.DataFrame(
                json.loads(
                    get_knmi_hourly_temperature_series(
                        self.time_stamps[0].tz_convert("Europe/Amsterdam").strftime("%Y%m%d%H"),
                        self.time_stamps[-1].tz_convert("Europe/Amsterdam").strftime("%Y%m%d%H"),
                    )
                )
            ).pipe(format_knmi)

        df = (
            cast(pl.Series, pl.from_pandas(self.time_stamps))
            .rename("datetime")
            .to_frame()
            .join(df.with_columns(pl.col("datetime").dt.cast_time_unit("ns")), on="datetime", how="left")
            .with_columns(pl.col("temperature").interpolate().forward_fill().backward_fill())
        )["temperature"]

        return cast(
            npt.NDArray[np.float64],
            df.to_pandas().values,
        )

    def area_solar_values(self, area: str) -> SolarProperties:
        """Get solar properties for a specific area.

        Args:
            area (str): Area identifier.

        Returns:
            SolarProperties object.
        """
        if self.area_latlon is None:
            return self._default_solar_values

        if area not in self._area_solar_values:
            try:
                self._area_solar_values[area] = SolarProperties(
                    time_stamps=self.time_stamps,
                    temperature=self.temperature,
                    latitude=self.area_latlon.loc[area, "latitude"],
                    longitude=self.area_latlon.loc[area, "longitude"],
                    config=self.config,
                )
            except KeyError:
                self._area_solar_values[area] = self._default_solar_values

        return self._area_solar_values[area]

    def optimize_parameters(self, net_load: pd.DataFrame) -> pd.DataFrame:
        """Optimize parameters for net_load."""
        logger.info("Optimizing parameters")

        f = self.time_stamps.isin(net_load.columns)

        logger.info("  - Create parameter array")
        params = [
            {
                "netload": row,
                "altitude": self.area_solar_values(i).altitude[f],
                "azimuth": self.area_solar_values(i).azimuth[f],
                "irradiance": self.area_solar_values(i).irradiance[f],
                "r_avg": self.area_solar_values(i).reference_average[f],
                "temperature": self.area_solar_values(i).temperature[f],
            }
            for i, row in net_load.iterrows()
        ]

        logger.info("  - Optimization")
        # scipy.minimize in _estimate_parameters already uses all threads, multiprocessing
        # on all cores will not help
        with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count() // 10), mp_context=MP_CONTEXT) as pool:
            result = pool.map(run_estimate_parameters, params)

        return pd.DataFrame(
            result, columns=["base", "effective_size", "tilt", "orientation", "c_temp"], index=net_load.index
        )

    def estimate_generation(self, net_load: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """Estimate generation based on net consumption and net generation.

        Args:
            net_load (DataFrame): DataFrame.
            result (DataFrame): Result.

        Returns:
            DataFrame with generation.
        """
        logger.info("Estimating generation...")
        if self.area_mapping is None:
            msg = "Area mapping is None, which should only happen during testing"
            raise ValueError(msg)

        param_df = result.copy()
        param_df["area"] = [self.area_mapping[idx] for idx in param_df.index]
        areas = param_df["area"].unique()
        with ProcessPoolExecutor(mp_context=MP_CONTEXT) as pool:
            logger.info("  - Area-dependent parameters")
            area_params = pd.DataFrame(
                list(pool.map(get_solar_properties, [self.area_solar_values(area) for area in areas])),
                index=areas,
                columns=["altitude", "azimuth", "irradiance", "temperature"],
            )
        param_df = param_df.join(area_params, on="area")

        logger.info("  - Customer-dependent generation")
        # Calculate generation for all customers
        sm = np.apply_along_axis(run_solar_maxgen_parallel, 1, param_df.iloc[:, 1:].values)

        logger.info("  - Final calculation")

        area_index = [self.area_mapping[idx] for idx in net_load.index]
        reference = self.pv_ratio.iloc[:, : net_load.shape[1]].reindex(index=area_index)
        reference.columns = self.time_stamps
        reference.index = net_load.index
        generation = reference * sm
        logger.info("Done estimating generation")

        return generation

    def calculate_simple_self_consumption(self, generation: pd.DataFrame, net_gen: pd.DataFrame) -> pd.DataFrame:
        """Calculate self consumption based on generation and net generation (export).

        Args:
            generation (DataFrame): Estimate generation.
            net_gen (DataFrame): Net generation (export).

        Returns:
            DataFrame with estimated self consumption.
        """
        logger.info("Calculate simple self consumption")
        self_consumption = generation - net_gen
        self_consumption[self_consumption < 0] = 0

        # Add MPI for simple self consumption
        self.calculate_simple_self_consumption_mpi(generation, net_gen, self_consumption)

        return self_consumption

    def _sum_grouped_by_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract year from date and sum by year."""
        grouped_df = df.T
        grouped_df["year"] = grouped_df.index.to_series().dt.year
        grouped_df = (
            grouped_df.groupby("year")
            .sum()
            .reset_index()
            .melt(id_vars="year", var_name="variable", value_name="value")
            .set_index(["variable", "year"])
        )

        return grouped_df

    def calculate_simple_self_consumption_mpi(
        self, generation: pd.DataFrame, net_gen: pd.DataFrame, self_consumption: pd.DataFrame
    ) -> None:
        """Calculate performance metrics based on simple self consumption.

        Args:
            generation (DataFrame): Estimated generation.
            net_gen (DataFrame): Net generation (export).
            self_consumption (DataFrame): Estimated simple self consumption.
        """
        diff = self._sum_grouped_by_year(generation - (self_consumption + net_gen))
        gen = self._sum_grouped_by_year(generation)
        self._mpi = ((gen - diff) / gen).rename(columns={"value": "self_consumption_mpi"})

    def calculate_self_consumption(
        self, generation: pd.DataFrame, net_con: pd.DataFrame, net_gen: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate average self consumption.

        Calculates the mean of the simple and the rule-based self consumption.

        Args:
            generation (DataFrame): Estimate generation.
            net_con (DataFrame): Net consumption (import).
            net_gen (DataFrame): Net consumption (export).

        Returns:
            DataFrame with estimated self consumption.
        """
        logger.info("Calculate self consumption")
        rule_based_self_consumption = calculate_rule_based_self_consumption(
            generation=generation, net_con=net_con, net_gen=net_gen
        )
        simple_self_consumption = self.calculate_simple_self_consumption(generation=generation, net_gen=net_gen)

        self_consumption = (rule_based_self_consumption + simple_self_consumption) / 2

        # Add MPI for self consumption
        self.calculate_self_consumption_mpi(self_consumption)

        return self_consumption

    def calculate_self_consumption_mpi(self, self_consumption: pd.DataFrame) -> None:
        """Calculate performance metric for self_consumption.

        Args:
            self_consumption (DataFrame): DataFrame with self_consumption.
        """
        tmp = self_consumption.T
        tmp["year"] = tmp.index.to_series().dt.year
        self._mpi = pd.concat(
            (
                self._mpi,
                tmp.groupby("year")
                .mean()
                .reset_index()
                .melt(id_vars="year", var_name="variable", value_name="mean_self_consumption")
                .set_index(["variable", "year"]),
                tmp.groupby("year")
                .max()
                .reset_index()
                .melt(id_vars="year", var_name="variable", value_name="max_self_consumption")
                .set_index(["variable", "year"]),
            ),
            axis=1,
        )

    def calculate_generation_consumption(
        self,
        generation: pd.DataFrame,
        net_con: pd.DataFrame,
        net_gen: pd.DataFrame,
        include_self_consumption: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate generation and consumption.

        When self_consumption is `False`, no self consumption is taken into account. When
        self_consumption is `True` (default), the estimated consumption is calculated based
        on the estimated self consumption

        Args:
            generation (DataFrame): Estimated generation.
            net_con (DataFrame): Net consumption (import).
            net_gen (DataFrame): Net generation (export).
            include_self_consumption (bool): Include estimate self consumption in
                calculation.

        Returns:
            Tuple with two DataFrames: generation and consumption.
        """
        logger.info("Calculation generation and consumption")
        if include_self_consumption is False:
            return generation, generation - (net_gen - net_con)

        self_consumption = self.calculate_self_consumption(generation=generation, net_con=net_con, net_gen=net_gen)
        generation = net_gen + self_consumption
        consumption = net_con + self_consumption
        return generation, consumption

    def _validate_area(self, net_load: pd.DataFrame) -> None:
        """Check that all areas in net_load are present in pv_ratio.

        Args:
            net_load (pd.DataFrame): DataFrame with the net_load, equivalent to
            net generation - net consumption.

        Raises:
            ValueError if at least one area is missing.
        """
        if self.area_mapping is None:
            return

        area_in_net_load = pd.Series([self.area_mapping[idx] for idx in net_load.index]).unique()
        areas_not_present = set(area_in_net_load) - set(self.pv_ratio.index)
        if len(areas_not_present) > 0:
            for area in areas_not_present:
                logger.error(f"Area {area} not present in pv_ratio!")

            msg = "One or more areas are not present in pv_ratio!"
            raise ValueError(msg)

    def disaggregate(
        self,
        net_con: pd.DataFrame | None = None,
        net_gen: pd.DataFrame | None = None,
        net_load: pd.DataFrame | None = None,
        include_self_consumption: bool = True,
        use_sampling: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Predict generation and consumption.

        Args:
            net_con (pd.DataFrame): DataFrame with the net consumption (import) values,
                these are the measured values from the smart meters. In the Dutch grid,
                this is also known as LDN ("levering door netbeheerder").
            net_gen (pd.DataFrame): DataFrame with the net generation (export) values,
                these are the measured values from the smart meters.In the Dutch grid,
                this is also known as ODN ("ontvangen door netbeheerder").
            net_load (pd.DataFrame): DataFrame with the net_load, equivalent to net
                generation - net consumption. Use this if net consumption and net
                generation are not separately available.
            include_self_consumption (bool, optional): Include estimated self consumption
                in the calculation.
            use_sampling (bool, optional): Run the parameter estimation on a subsampled
                time series. Disaggregation will still be performed on the full series.
                Default is `False`.

        Returns:
            Dictionary with two keys, generation and consumption. Values are DataFrames.
        """
        # Create net consumption, generation and load from (partial) input
        net_con, net_gen, net_load = self._calculate_load(net_con, net_gen, net_load)

        # Validate pv_ratio
        self._validate_area(net_load=net_load)

        # Determine installation-specific PV parameters
        if use_sampling:
            optimize_data = net_load.loc[:, (net_load.columns.hour > 4) & (net_load.columns.hour < 23)]
            if optimize_data.shape[1] > N_SAMPLE:
                optimize_data = optimize_data.sample(N_SAMPLE, axis=1).sort_index(axis=1)

            params = self.optimize_parameters(optimize_data)
        else:
            params = self.optimize_parameters(net_load)

        # Calculate generation and consumption
        generation = self.estimate_generation(net_load, params)
        generation = correct_generation_with_net_gen(generation=generation, net_gen=net_gen)
        generation, consumption = self.calculate_generation_consumption(
            generation=generation, net_gen=net_gen, net_con=net_con, include_self_consumption=include_self_consumption
        )
        consumption = generation - net_load

        logger.info("Done with disaggregation")

        return {
            "params": params,
            "net_con": net_con,
            "net_gen": net_gen,
            "generation": generation,
            "consumption": consumption,
        }

    def _calculate_load(
        self, net_con: pd.DataFrame | None, net_gen: pd.DataFrame | None, net_load: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Calculate net consumption, generation and load."""
        if net_con is None and net_gen is None and net_load is None:
            msg = "Need either net_con and net_gen as input, or net_load."
            raise ValueError(msg)

        if net_load is None and net_con is not None and net_gen is not None:
            net_gen = net_gen.fillna(0)
            net_con = net_con.fillna(0)
            net_load = net_gen - net_con
        elif net_load is not None and net_con is None and net_gen is None:
            net_con = pd.DataFrame(0.0, index=net_load.index, columns=net_load.columns)
            net_gen = pd.DataFrame(0.0, index=net_load.index, columns=net_load.columns)
            net_gen[net_load > 0] = net_load[net_load > 0]
            net_con[net_load < 0] = -net_load[net_load < 0]
        else:
            # all specified
            msg = "Need either net_con and net_gen as input, or net_load, not all of them."
            raise ValueError(msg)

        if len(self.time_stamps) == 0:
            logger.info("Using time_stamps from net load.")
            self.time_stamps = net_load.columns
            # No reference was supplied, so temperature could not be determined yet.
            # Now that the time stamps are known, derive it from the data/KNMI.
            if len(self.temperature) == 0:
                self.temperature = None
        elif len(self.time_stamps) != len(net_load.columns):
            msg = "net load needs the same number of measurements as the provided PV reference!"
            raise ValueError(msg)

        # Make sure columns are time stamps
        net_load.columns = self.time_stamps
        net_gen.columns = self.time_stamps
        net_con.columns = self.time_stamps

        return net_con, net_gen, net_load


def correct_generation_with_net_gen(generation: pd.DataFrame, net_gen: pd.DataFrame) -> pd.DataFrame:
    """Correct disaggregated generation based on net_gen.

    Performs two correction:
        * If estimated generation is 0, the net_gen value will be used (if higher than 0).
        * If the maximum of the net_gen is higher than that of the generation, all
            generation values will be scaled to maximum(net_gen)/maximum(generation).

    Args:
        generation (DataFrame): Generation from disaggregation.
        net_gen (DataFrame): Net generation (export).

    Returns:
        Corrected generation.
    """
    max_net_gen = net_gen.max(1).values
    max_generation = generation.max(1).values

    max_gen_is_zero = max_generation == 0
    max_net_gen_higher = max_net_gen > max_generation

    corrected = generation.values.copy()

    # Perform correction
    corrected[max_net_gen_higher & max_gen_is_zero, :] = net_gen.values[max_net_gen_higher & max_gen_is_zero, :]
    scale = np.ones_like(max_generation, dtype=np.float64)
    np.divide(max_net_gen, max_generation, out=scale, where=max_generation != 0)

    rows_to_scale = max_net_gen_higher & ~max_gen_is_zero
    corrected[rows_to_scale, :] = corrected[rows_to_scale, :] * scale[rows_to_scale, np.newaxis]

    return pd.DataFrame(corrected, index=generation.index, columns=generation.columns)


@njit(parallel=True)  # type:ignore
def apply_generation_rules(
    generation: npt.NDArray[np.float64],
    net_gen: npt.NDArray[np.float64],
    net_con: npt.NDArray[np.float64],
    mean_net_con: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculate self consumption based on decision tree.

    The self consumption is determined based on the estimated generation in
    combination with net consumption and net generation. Depending on different
    conditions, self consumption is estimated.

    Args:
        generation (array): Estimate generation.
        net_con (array): Net consumption (import).
        net_gen (array): Net generation (export).
        mean_net_con (array): Mean net generation per customer, based on all non-zero values.

    Returns:
        array with estimated self consumption.
    """
    self_consumption = generation.copy()
    nrows, ncols = np.shape(self_consumption)
    for i in prange(nrows):
        for j in range(ncols):
            should_be_zero = (
                (self_consumption[i][j] == 0)
                | (self_consumption[i][j] < net_gen[i][j])
                | ((net_con[i][j] == 0) & (net_gen[i][j] == 0))
            )
            if should_be_zero:
                self_consumption[i][j] = 0.0
            elif (net_con[i][j] == 0) & (net_gen[i][j] > 0):  # net_gen_only
                self_consumption[i][j] = np.minimum(self_consumption[i][j] - net_gen[i][j], mean_net_con[i])
            elif (net_con[i][j] > 0) & (net_gen[i][j] == 0):  # net_con_only
                self_consumption[i][j] = np.minimum(self_consumption[i][j], net_con[i][j])
            elif (net_con[i][j] > net_gen[i][j]) & (net_gen[i][j] != 0):  # net_con_gt_net_gen
                self_consumption[i][j] = np.minimum(self_consumption[i][j] - net_gen[i][j], net_gen[i][j])
            elif (net_con[i][j] <= net_gen[i][j]) & (net_con[i][j] != 0):  # net_con_le_net_gen
                self_consumption[i][j] = np.minimum(self_consumption[i][j] - net_gen[i][j], net_con[i][j])

    return self_consumption


def calculate_rule_based_self_consumption(
    generation: pd.DataFrame, net_con: pd.DataFrame, net_gen: pd.DataFrame
) -> pd.DataFrame:
    """Calculate self consumption based on decision tree.

    The self consumption is determined based on the estimated generation in
    combination with net consumption and net generation. Depending on different
    conditions, self consumption is estimated.

    Args:
        generation (DataFrame): Estimate generation.
        net_con (array): Net consumption (import).
        net_gen (DataFrame): Net generation (export).

    Returns:
        DataFrame with estimated self consumption.
    """
    logger.info("Calculate rule-based self consumption")

    # switch to numpy as pandas is very slow here
    idx = generation.index
    columns = generation.columns

    # Determine mean net consumption per customer for all non-zero measurements
    mean_net_con = np.nan_to_num(mean_gt_zero(net_con, axis=1))

    # Apply the rules based on fast Numba function
    self_consumption = apply_generation_rules(generation.values, net_gen.values, net_con.values, mean_net_con)

    return pd.DataFrame(self_consumption, index=idx, columns=columns)


def run_solar_maxgen_parallel(args: Any) -> np.typing.NDArray[np.float64]:
    """Run solar_maxgen based on tuple of parameters."""
    (effective_size, tilt, orientation, c_temp, area, altitude, azimuth, irradiance, temperature) = args

    maxgen = solar_maxgen_parallel(
        cos_altitude=np.cos(altitude),
        sin_altitude=np.sin(altitude),
        azimuth=azimuth,
        irradiance=irradiance,
        effective_size=effective_size,
        tilt=tilt,
        orientation=orientation,
        temperature=temperature,
        c_temp=c_temp,
    )

    return cast(np.typing.NDArray[np.float64], maxgen)


def run_estimate_parameters(kwargs: Any) -> np.typing.NDArray[np.float64]:
    """Run estimate_parameters based on dictionary of parameters."""
    return cast(np.typing.NDArray[np.float64], estimate_parameters(**kwargs))


def get_solar_properties(
    solar_prop: SolarProperties,
) -> tuple[
    np.typing.NDArray[np.float64],
    np.typing.NDArray[np.float64],
    np.typing.NDArray[np.float64],
    np.typing.NDArray[np.float64],
]:
    """Return altitude, azimuth, irradiance and temperature from SolarProperties object."""
    return (
        solar_prop.altitude,
        solar_prop.azimuth,
        solar_prop.irradiance,
        solar_prop.temperature,
    )
