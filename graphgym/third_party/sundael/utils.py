# SPDX-FileCopyrightText: 2026 Contributors to the sundael project.
# SPDX-License-Identifier: MPL-2.0
"""Utility functions."""

from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import njit


@njit  # type: ignore
def np_apply_along_axis(
    func1d: Callable[[np.typing.NDArray[Any]], np.typing.NDArray[Any]],
    axis: Literal[0, 1],
    arr: np.typing.NDArray[Any],
) -> np.typing.NDArray[Any]:
    """Use axis keyword in numba-optimized functions."""
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@njit  # type: ignore
def np_min(array: np.typing.NDArray[Any], axis: Literal[0, 1]) -> Any:  # pragma: nocover
    """min() function with axis argument, for use with numba.

    Numba doesn't support optional argument for np.min().
    """
    return np_apply_along_axis(np.min, axis, array)


@njit  # type: ignore
def np_max(array: np.typing.NDArray[Any], axis: Literal[0, 1]) -> Any:  # pragma: nocover
    """max() function with axis argument, for use with numba.

    Numba doesn't support optional argument for np.min().
    """
    return np_apply_along_axis(np.max, axis, array)


@njit  # type: ignore
def np_nanmean(array: np.typing.NDArray[Any], axis: Literal[0, 1]) -> Any:  # pragma: nocover
    """nanmean() function with axis argument, for use with numba.

    Numba doesn't support optional argument for np.min().
    """
    return np_apply_along_axis(np.nanmean, axis, array)


def mean_gt_zero(df: pd.DataFrame | npt.NDArray[np.float64], axis: Literal[0, 1] = 0) -> npt.NDArray[np.float64]:
    """Calculate mean of all values > 0.

    Args:
        df (DataFrame): Input DataFrame.
        axis: Either 0 or 1.

    Returns:
        Series with mean.
    """
    _df = df.astype(float)
    _df[_df <= 0.0] = np.nan
    valid_counts = np.sum(~np.isnan(_df), axis=axis)
    summed = np.nansum(_df, axis=axis)
    means = np.full(np.shape(summed), np.nan, dtype=np.float64)
    # Store results in-place in means
    np.divide(summed, valid_counts, out=means, where=valid_counts > 0)
    return cast(npt.NDArray[np.float64], means)
