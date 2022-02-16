#!/usr/bin/env python3

"""
Postprocessing calculations for swept power data.
"""

import os

os.environ["NUMEXPR_MAX_THREADS"] = "4"

import numpy as np
import pandas as pd
import numexpr as ne
import warnings
warnings.filterwarnings("ignore", message="divide by zero")
warnings.filterwarnings("ignore", message="invalid value encountered")


def dBtopow(x):
    """Computes `10**(x/10.)` with speed optimizations"""
    # for large arrays, this is much faster than just writing the expression in python
    values = ne.evaluate("10**(x/10.)", local_dict=dict(x=x))

    if isinstance(x, pd.Series):
        return pd.Series(values, index=x.index)
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(values, index=x.index, columns=x.columns)
    else:
        return values


def powtodB(x, abs: bool = True, eps: float = 0):
    """Computes `10*log10(abs(x) + eps)` or `10*log10(x + eps)` with speed optimizations"""

    # for large arrays, this is much faster than just writing the expression in python
    eps_str = "" if eps == 0 else "+eps"

    if abs:
        values = ne.evaluate(
            f"10*log10(abs(x){eps_str})", local_dict=dict(x=x, eps=eps)
        )
    else:
        values = ne.evaluate(f"10*log10(x+eps){eps_str}", local_dict=dict(x=x, eps=eps))

    if isinstance(x, pd.Series):
        return pd.Series(values, index=x.index)
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(values, index=x.index, columns=x.columns)
    else:
        return values


def envtopow(x):
    """Computes abs(x)**2 with speed optimizations"""
    values = ne.evaluate("abs(x)**2", local_dict=dict(x=x))

    if np.iscomplexobj(values):
        values = np.real(values)

    if isinstance(x, pd.Series):
        return pd.Series(values, index=x.index)
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(values, index=x.index, columns=x.columns)
    else:
        return values


def envtodB(x, abs: bool = True, eps: float = 0):
    """Computes `20*log10(abs(x) + eps)` or `20*log10(x + eps)` with speed optimizations"""
    # for large arrays, this is much faster than just writing the expression in python
    eps_str = "" if eps == 0 else "+eps"

    if abs:
        values = ne.evaluate(
            f"20*log10(abs(x){eps_str})", local_dict=dict(x=x, eps=eps)
        )
    else:
        values = ne.evaluate(
            f"20*log10(x+eps){eps_str}", local_dict=dict(x=x, eps=eps)
        )

    if np.iscomplexobj(values):
        values = np.real(values)

    if isinstance(x, pd.Series):
        return pd.Series(values, index=x.index)
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(values, index=x.index, columns=x.columns)
    else:
        return values


def iq_to_bin_power(iq: np.array, Ts: float, Tbin: float):
    """returns average power along the rows of `iq` (time axis) averaged by bins of duration Tbin"""
    if not np.isclose(Tbin % Ts, 0, atol=1e-6):
        print(Tbin % Ts)
        raise ValueError(
            f"bin period ({Tbin} s) must be multiple of waveform sample period ({Ts})"
        )

    N = int(Tbin / Ts)

    pow = envtopow(iq)

    # truncate to integer number of bins
    pow = pow[: (iq.shape[0] // N) * N]

    pow = (
        pow
        # insert a new axis with bin size N
        .reshape(pow.shape[0] // N, N, pow.shape[1])
        # mean along the bin axis
        .mean(axis=1)
    )

    Nmax = min(pow.shape[0], iq.shape[0] // N)

    return pow[:Nmax]


def unstack_to_2d_blocks(pvt: pd.Series, Tblock: float) -> pd.DataFrame:
    """unstack time series of power vs time (time axis) `pvt` into
    a pd.DataFrame in which row consists of time series of time duration `Twindow`.

    Arguments:

        pvt: indexed by TimedeltaIndex or TimeIndex

        Tblock: time duration of the block

    """

    Tbin = pvt.index[1] - pvt.index[0]

    if np.isclose(Tblock % Tbin, 0, 1e-6):
        print(Tblock, Tbin, Tblock % Tbin)
        raise ValueError(
            "analysis window length must be multiple of the power INTEGRATION length"
        )

    N = int(Tblock / Tbin)

    pvt = pvt.iloc[: N * (pvt.shape[0] // N)]

    values = (
        pvt.values
        # insert a new axis with bin size N
        .reshape(pvt.shape[0] // N, N)
    )

    df = pd.DataFrame(values, index=pvt.index[::N], columns=pvt.index[:N])

    df.columns.name = "Analysis window time elapsed (s)"
    df.index = pd.TimedeltaIndex(df.index, unit="s")

    return df


def hist_laxis(x: np.ndarray, n_bins: int, range_limits: tuple) -> np.ndarray:
    """Computes a histogram along the last axis of an input array.

    For reference see https://stackoverflow.com/questions/44152436/calculate-histograms-along-axis

    Args:
        x: input data of shape (M[0], ..., M[K-1], N)

        n_bins: Number of bins in the histogram

        range_limits: Bounds on the histogram bins [lower bound, upper bound] inclusive

    Returns:
        np.ndarray of shape (M[0], ..., M[K-1], n_bins)
    """

    # Setup bins and determine the bin location for each element for the bins
    N = x.shape[-1]
    bins = np.linspace(range_limits[0], range_limits[1], n_bins + 1)
    data2D = x.reshape(-1, N)
    idx = np.searchsorted(bins, data2D, "right") - 1

    # Some elements would be off limits, so get a mask for those
    bad_mask = (idx == -1) | (idx == n_bins)

    # We need to use bincount to get bin based counts. To have unique IDs for
    # each row and not get confused by the ones from other rows, we need to
    # offset each row by a scale (using row length for this).
    scaled_idx = n_bins * np.arange(data2D.shape[0])[:, None] + idx

    # Set the bad ones to be last possible index+1 : n_bins*data2D.shape[0]
    limit = n_bins * data2D.shape[0]
    scaled_idx[bad_mask] = limit

    # Get the counts and reshape to multi-dim
    counts = np.bincount(scaled_idx.ravel(), minlength=limit + 1)[:-1]
    counts.shape = x.shape[:-1] + (n_bins,)
    return counts


def power_histogram_along_axis(
    pvt: pd.DataFrame,
    bounds: tuple((float, float)),
    resolution_db: float,
    resolution_axis: int = 1,
    truncate: bin = True,
    dtype="uint32",
    axis=0
) -> pd.DataFrame:

    """Computes a histogram along the time index of a pd.Series time series of power readings.

    Args:
        pvt: a pd.Series or 1-column pd.DataFrame of power levels in linear units

        bounds: [lower, upper] bounds for the histogram power level bins (upper-bound inclusive)

        resolution_db: step size of the histogram power level bins

        resolution_axis: number of indices to group into a single time bin

        truncate: whether to truncate pvt to an integer factor of `resolution_axis`

        dtype: the integer data type to return for histogram counts

    Returns:
        `pd.DataFrame` instance, indexed on time, columned by power transformed into dB, with values of type `dtype`

    Raises:
        ValueError: if not truncate and len(pvt) % resolution_axis != 0

    """

    if axis == 0:
        pvt = pvt.T
    elif axis != 1:
        raise ValueError(f'axis argument must be 0 or 1')

    # truncate to an integer number of sweep blocks
    pvt = powtodB(pvt, abs=False)

    if not truncate and len(pvt) % resolution_axis != 0:
        raise ValueError(
            "non-integer number of sweeps in pvt; pass truncate=False to truncate"
        )

    pvt = pvt.iloc[: resolution_axis * (len(pvt) // resolution_axis)]

    # use hist_laxis to compute the histogram at each time point
    shape = pvt.shape[0] // resolution_axis, pvt.shape[1] * resolution_axis
    reshaped = pvt.values.reshape(shape)
    n_bins = 1 + int((bounds[1] - bounds[0]) / resolution_db)
    h = hist_laxis(reshaped, n_bins, bounds).astype(dtype)

    # pack a DataFrame with the bin labels
    #     timestamps = pvt.index.get_level_values('Time')
    #     time_bins = pd.to_datetime(timestamps[::resolution_axis])
    power_bins = np.linspace(bounds[0], bounds[1], n_bins).astype("float64")
    df = pd.DataFrame(h, index=pvt.index[::resolution_axis], columns=power_bins)

    return df
