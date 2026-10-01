# SPDX-FileCopyrightText: 2026 Contributors to the sundael project.
# SPDX-License-Identifier: MPL-2.0
"""Optimization of parameters of solar installations."""

from functools import cached_property
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pvlib
from numba import njit, prange
from scipy.optimize import minimize

from sundael.config import (
    DEFAULT_CONFIG,
    PVConfig,
)


class SolarProperties:
    """Solar properties class."""

    def __init__(
        self,
        time_stamps: pd.Series,
        latitude: float | None = None,
        longitude: float | None = None,
        temperature: float | pd.Series | None = None,
        config: PVConfig | None = None,
    ) -> None:
        """Solar properties based on location and time.

        Properties are calculated based on time, location (latitude and longitude)
        and temperature.

        Args:
            time_stamps (Series): Series of pandas datetime values.
            latitude (float, optional): Latitude of location.
            longitude (float, optional): Longitude of location.
            temperature (float or Series, optional): Temperature, either the average
                temperature as float, or the temperature per timepoint.
            config (PVConfig, optional): Configuration object with default values.
        """
        self.config = config if config is not None else DEFAULT_CONFIG
        self.time_stamps = time_stamps
        self.latitude = self.config.latitude if latitude is None else latitude
        self.longitude = self.config.longitude if longitude is None else longitude
        if temperature is None:
            temperature = self.config.yearly_avg_temp
        self.temperature = np.array([temperature] * len(time_stamps)) if isinstance(temperature, float) else temperature

    @cached_property
    def position(self) -> pd.DataFrame:
        """DataFrame with location columns."""
        loc = pvlib.location.Location(latitude=self.latitude, longitude=self.longitude, tz="UTC")
        p = loc.get_solarposition(times=self.time_stamps)

        # signed range vs. positive range
        p["azimuth"] = p["azimuth"] - 180

        # convert degrees to radians
        p = p[["zenith", "apparent_zenith", "elevation", "apparent_elevation", "azimuth"]] * np.pi / 180

        # Add air mass
        p = p.join(loc.get_airmass(times=self.time_stamps))

        return p

    @cached_property
    def azimuth(self) -> npt.NDArray[np.float64]:
        """array: Sun azimuth for each timestamp."""
        return cast(npt.NDArray[np.float64], self.position["azimuth"].values)

    @cached_property
    def altitude(self) -> npt.NDArray[np.float64]:
        """array: Sun altitude for each timestamp."""
        return cast(npt.NDArray[np.float64], self.position["apparent_elevation"].values)

    @cached_property
    def air_mass(self) -> npt.NDArray[np.float64]:
        """array: Air mass for each timestamp."""
        return cast(npt.NDArray[np.float64], self.position["airmass_absolute"].values)

    @cached_property
    def irradiance(self) -> npt.NDArray[np.float64]:
        """Intensity on a plane perpendicular to the sun's rays in units of kW/m2.

        Formula according to Meinel & Meinel, 1976.
        See: https://www.pveducation.org/pvcdrom/properties-of-sunlight/air-mass
        """
        # - 1.353 kW/m2 is the solar constant
        # - the number 0.7 arises from the fact that about 70% of the radiation incident
        #   on the atmosphere is transmitted to the Earth
        # - The extra power term of 0.678 is an empirical fit to the observed data and takes
        #   into account the non-uniformities in the atmospheric layers.
        irradiance = 1.353 * 0.7 ** (np.sign(self.air_mass) * np.abs(self.air_mass) ** 0.678)
        irradiance[np.isnan(irradiance)] = 0
        return irradiance

    @cached_property
    def reference_average(self) -> npt.NDArray[np.float64]:
        """Calculate average theoretical generation for reference."""
        return cast(
            npt.NDArray[np.float64],
            solar_maxgen(
                cos_altitude=np.cos(self.altitude),
                sin_altitude=np.sin(self.altitude),
                azimuth=self.azimuth,
                irradiance=self.irradiance,
                effective_size=1,
                tilt=self.config.default_tilt,
                orientation=self.config.default_orientation,
                c_temp=self.config.default_c_temp,
                temperature=self.temperature,
            ),
        )


def solar_maxgen_core(
    cos_altitude: npt.NDArray[np.float64],
    sin_altitude: npt.NDArray[np.float64],
    azimuth: npt.NDArray[np.float64],
    irradiance: npt.NDArray[np.float64],
    effective_size: float,
    tilt: float,
    orientation: float,
    temperature: npt.NDArray[np.float64] | float,
    c_temp: float,
) -> Any:
    """Calculate theoretical maximum irradiance at PV panel.

    Implements the algorithm from the following article:
      SunDance: Black-box Behind-the-Meter Solar Disaggregation.
      Chen & Irwin, 2017, https://dl.acm.org/doi/10.1145/3077839.3077848

    Args:
        cos_altitude (array): Cosinue of sun altitude.
        sin_altitude (array): Sine of sun altitude.
        azimuth (array): Sun azimth.
        irradiance (array): Irridiance.
        effective_size (float): Sizes and efficiencies of the PV installation.
        tilt (float): Tilt of PV panel, 0 = horizontal, pi/2 = vertical, 0.65 = ideal (37 degrees).
        orientation (float): Orientation of panel (north/east/south/west), 0 = south (ideal),
            pi/2 = west, pi = north, -pi/2 = east
        temperature (float or array): Temperature in degrees Celsius.
        c_temp (float): Coefficient for the temperature dependency.


    Returns:
      float: maximum irradiance.
    """
    t_baseline = 0
    n = len(irradiance)

    solar = np.zeros(n)
    if isinstance(temperature, (float, int)):
        temperature = np.array([temperature] * n)

    sin_tilt = np.sin(tilt)
    cos_tilt = np.cos(tilt)
    c_temp_base = c_temp / 100.0

    for i in prange(n):
        if sin_altitude[i] < 0:
            continue

        t_adj_effective_size = effective_size * (1.0 + c_temp_base * (t_baseline - temperature[i]))

        # Theoretical received irradiance at PV panel (not adjusted for weather)
        solar[i] = (
            irradiance[i]
            * t_adj_effective_size
            * (
                (cos_altitude[i] * sin_tilt * np.cos(orientation - azimuth[i]) + sin_altitude[i] * cos_tilt)
                # Light is scattered by the atmosphere, of which some is received by PV panel.
                # Diffuse irradiance (estimated at ~10%, does not depend on angles)
                + 0.1
            )
        )

    return solar


# Non-parallel version that can be used in scipy.minimize. If a parallel version
# is used, it clashes with the parallization in minimize, resulting in extreme
# slowdown.
solar_maxgen = njit(solar_maxgen_core, fastmath=True)

# Parallel version that is, by itself, much faster than the non-parallel version.
solar_maxgen_parallel = njit(solar_maxgen_core, fastmath=True, parallel=True)


@njit(fastmath=True)  # type: ignore
def algf(
    params: tuple[np.float64, np.float64, np.float64, np.float64, np.float64],
    w: npt.NDArray[np.float64],
    netload: npt.NDArray[np.float64],
    n_time_stamps: int,
    error: float,
    cos_altitude: npt.NDArray[np.float64],
    sin_altitude: npt.NDArray[np.float64],
    azimuth: npt.NDArray[np.float64],
    irradiance: npt.NDArray[np.float64],
    temperature: npt.NDArray[np.float64] | float,
) -> Any:
    """Loss function to optimize.

    Calculates the (mathematical) loss between estimated gross generation and measured gross
    generation of reference.

    Args:
        params (tuple): List with customer specific parameters
        w (np.array): Average theoretical generation of reference (only timestamps
            where sign is the same as for netload)
        netload (np.array): Net load (net generation - net consumption).
        n_time_stamps (int): Number of time stamps.
        error (float): Penalty value, extra penalty when diff is > 0.
        cos_altitude (array): Cosine of sun altitude
        sin_altitude (array): Sine of sun altitude
        azimuth (array): Sun azimuth
        irradiance (array): Irridiance
        temperature (float or array): Temperature in degrees Celsius.

    Returns:
        float: Loss of estimated gross generation.
    """
    base, effective_size, tilt, orientation, c_temp = params

    # For this customer: net load minus maximum generation plus consumption of PV converter
    diffs = (
        netload
        + base
        - solar_maxgen(
            cos_altitude,
            sin_altitude,
            azimuth,
            irradiance,
            effective_size,
            tilt,
            orientation,
            temperature=temperature,
            c_temp=c_temp,
        )
    )

    diffs[diffs > 0] = error * diffs[diffs > 0]

    return np.sum(np.abs(w * diffs)) / n_time_stamps


def estimate_parameters(
    netload: pd.Series,
    altitude: np.typing.NDArray[np.float64],
    azimuth: np.typing.NDArray[np.float64],
    irradiance: np.typing.NDArray[np.float64],
    r_avg: np.typing.NDArray[np.float64],
    temperature: np.typing.NDArray[np.float64] | float,
) -> np.typing.NDArray[np.float64]:
    """Parameter optimization function.

    Args:
        netload (pd.Series): Netload.
        altitude (np.array): Sun altitude.
        azimuth (np.array):  Sun azimuth.
        irradiance (np.array): Irridiance.
        r_avg (np.array): Average theoretical generation for reference.
        temperature (float or array): Temperature in degrees Celsius.

    Returns:
        Array with estimated values for base, effective_size, tilt and orientation.
    """
    n_time_stamps = netload.shape[0]

    # Select timestamps for which sign of reference and netload is the same
    w = np.zeros(n_time_stamps)
    idx = np.sign(r_avg) == np.sign(netload)
    w[idx] = r_avg[idx]

    # Initial parameters
    # initial_guess = (0, np.max(netload) / np.max(r_avg), 0.65, 0)
    initial_guess = (0, np.max(netload) / np.max(r_avg), 0.4, -0.1, 0.5)  # 5-10% faster

    # Parameter bounds
    bounds = [
        (0, 100_000),
        (0, 100_000),
        (0.1, 1.5),
        (-1.5, 1.5),
        (0, 2),
    ]

    # Constant arguments to optimization function
    args = (w, netload.values, n_time_stamps, 100, np.cos(altitude), np.sin(altitude), azimuth, irradiance, temperature)

    # Run the optimization
    result = minimize(algf, initial_guess, args=args, method="L-BFGS-B", bounds=bounds)

    return cast(np.typing.NDArray[np.float64], result.x)
