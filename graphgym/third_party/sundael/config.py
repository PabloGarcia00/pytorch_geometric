# SPDX-FileCopyrightText: 2026 Contributors to the sundael project.
# SPDX-License-Identifier: MPL-2.0
"""Configuration of default parameters."""

from pydantic import BaseModel, ConfigDict, Field


class PVConfig(BaseModel):
    """Default configuration values for the package.

    Users can override any field by creating a new ``PVConfig`` instance.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    latitude: float = Field(
        default=52.3,
        description=(
            "Default latitude to use. This is replaced by the area_latlon if it is provided to PVDisaggregator. "
            "The default latitude and longitude point towards the Netherlands."
        ),
    )
    longitude: float = Field(
        default=4.7,
        description=(
            "Default longitude to use. This is replaced by the area_latlon if it is provided to PVDisaggregator. "
            "The default latitude and longitude point towards the Netherlands."
        ),
    )
    auto_infer_temp: bool = Field(
        default=True,
        description=(
            "Use data from the Dutch meteorological institute (KNMI) to infer temperature. Set to `False` "
            "to disable for other localities."
        ),
    )

    yearly_avg_temp: float = Field(
        default=11.3,
        description=(
            "Default temperature. This is replaced by temperature if it is provided to PVDisaggregator. "
            "This default temperature is the average temperature of Netherlands from https://www.knmi.nl/klimaat."
        ),
    )

    # The constants below should probably not be changed
    default_tilt: float = 0.65
    default_orientation: float = 0
    default_c_temp: float = 0.5


DEFAULT_CONFIG = PVConfig()
