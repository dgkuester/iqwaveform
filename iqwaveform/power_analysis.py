""" Transformations and statistical tools for power data """

import os

os.environ.setdefault("NUMEXPR_MAX_THREADS", "4")

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
        values = ne.evaluate(f"20*log10(x+eps){eps_str}", local_dict=dict(x=x, eps=eps))

    if np.iscomplexobj(values):
        values = np.real(values)

    if isinstance(x, pd.Series):
        return pd.Series(values, index=x.index)
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(values, index=x.index, columns=x.columns)
    else:
        return values


def iq_to_bin_power(
    iq: np.array,
    Ts: float,
    Tbin: float,
    randomize: bool = False,
    kind: str = "mean",
    truncate=False,
):
    """computes power along the rows of `iq` (time axis) on bins of duration Tbin.

    Args:
        iq: complex-valued input waveform samples
        Ts: sample period of the input waveform
        Tbin: time duration of the bin size
        randomize: if True, randomize the start locations of the bins; otherwise, bins are contiguous
        kind: the detection operation in each bin, one of 'min', 'max', 'median', or 'mean'
        truncate: if True, truncate the last samples of `iq` to an integer number of bins
    """

    VALID_DETECTORS = ("min", "max", "median", "mean")

    if not truncate and not np.isclose(Tbin % Ts, 0, atol=1e-6):
        raise ValueError(
            f"bin period ({Tbin} s) must be multiple of waveform sample period ({Ts})"
        )

    if kind not in VALID_DETECTORS:
        raise ValueError(f"kind argument must be one of {VALID_DETECTORS}")

    N = int(Tbin / Ts)

    # instantaneous power, reshaped into bins
    if randomize:
        starts = np.random.randint(
            low=0, high=iq.shape[0] - N, size=int(np.rint(iq.shape[0] / N))
        )
        offsets = np.arange(N)

        power_bins = envtopow(iq)[starts[:, np.newaxis] + offsets[np.newaxis, :]]

    else:
        iq = iq[: (iq.shape[0] // N) * N]
        power_bins = envtopow(iq).reshape(
            (iq.shape[0] // N, N) + tuple([iq.shape[1]] if iq.ndim == 2 else [])
        )

    detector_ufunc = getattr(np, kind)
    return detector_ufunc(power_bins, axis=1)

    Nmax = min(pow.shape[0], iq.shape[0] // N)
    return pow[:Nmax]


def iq_to_cyclic_power(
    iq: np.ndarray, Ts: float, detector_period: float, frame_period: float, truncate=False
) -> dict:
    """computes a time series of periodic frame power statistics.

    The time axis on the frame time elapsed spans [0, frame_period) binned with step size
    `detector_period`, for a total of `int(frame_period/detector_period)` samples.

    RMS and peak power detector data are returned. For each type of detector, a time
    series is returned for (min, mean, max) statistics, which are computed across the
    number of frames (`frame_period/Ts`).

    Args:
        iq: complex-valued input waveform samples
        Ts: sample period of the iq waveform
        detector_period: sampling period within the frame
        frame_period: frame period to analyze

    Raises:
        ValueError: if detector_period%Ts != 0 or frame_period%detector_period != 0

    Returns:
        dict keyed on ('rms', 'peak') with values (min: np.array, mean: np.array, max: np.array)
    """
    if not np.isclose(frame_period % Ts, 0, atol=Ts/4):
        raise ValueError(
            "frame period must be positive integer multiple of the sampling period"
        )

    if not np.isclose(detector_period % Ts, 0, atol=Ts/4):
        raise ValueError(
            "detector_period period must be positive integer multiple of the sampling period"
        )

    frame_samples = int(np.rint(frame_period / Ts))
    frame_detector_bins = int(np.rint(frame_period / detector_period))

    if iq.shape[0] % frame_samples != 0:
        if truncate:
            iq = iq[:(iq.shape[0]//frame_samples)*frame_samples]
        else:
            raise ValueError("pass truncate=True to allow truncation to integer number of cyclic periods")

    # set up dimensions to make the statistics fast
    chunked_shape = (
        iq.shape[0] // frame_samples,
        frame_detector_bins,
        frame_samples // frame_detector_bins,
        *([iq.shape[1]] if iq.ndim == 2 else [])
    )
    iq_bins = iq.reshape(chunked_shape)

    power_bins = envtopow(iq_bins)

    # first, apply the detector statistic
    rms_power = power_bins.mean(axis=2)
    peak_power = power_bins.max(axis=2)

    # then, the cycle statistic
    return {
        "rms": (rms_power.min(axis=0), rms_power.mean(axis=0), rms_power.max(axis=0)),
        "peak": (
            peak_power.min(axis=0),
            peak_power.mean(axis=0),
            peak_power.max(axis=0),
        ),
    }


def iq_to_frame_power(
    iq: np.ndarray, Ts: float, detector_period: float, frame_period: float, truncate=False
) -> dict:

    warnings.warn('iq_to_frame_power has been deprecated. use iq_to_cyclic_power instead')

    return iq_to_cyclic_power(**locals())


def unstack_series_to_bins(
    pvt: pd.Series, Tbin: float, truncate: bool = False
) -> pd.DataFrame:
    """unstack time series of power vs time (time axis) `pvt` into
    a pd.DataFrame in which row consists of time series of time duration `Twindow`.

    Arguments:

        pvt: indexed by TimedeltaIndex or TimeIndex

        Tblock: time duration of the block

    """

    Ts = pvt.index[1] - pvt.index[0]

    if not truncate and not np.isclose(Tbin % Ts, 0, 1e-6):
        print(Tbin, Ts, Tbin % Ts)
        raise ValueError(
            "analysis window length must be multiple of the power INTEGRATION length"
        )

    N = int(np.rint(Tbin / Ts))

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


def sample_ccdf(a: np.array, edges: np.array, density: bool = True) -> np.array:
    """computes the fraction (or total number) of samples in `a` that
    exceed each edge value.

    Args:
        a: the vector of input samples
        edges: sample threshold values at which to characterize the distribution
        density: if True, the sample counts are normalized by `a.size`

    Returns:
        the empirical complementary cumulative distribution
    """

    # 'left' makes the bin interval open-ended on the left side
    # (the CCDF is "number of samples exceeding interval", and not equal to)
    edge_inds = np.searchsorted(edges, a, side="left")
    bin_counts = np.bincount(edge_inds, minlength=edges.size + 1)
    ccdf = (a.size - bin_counts.cumsum())[:-1]

    if density:
        ccdf = ccdf.astype("float64")
        ccdf /= a.size

    return ccdf


def hist_laxis(x: np.ndarray, n_bins: int, range_limits: tuple) -> np.ndarray:
    """computes a histogram along the last axis of an input array.

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
    axis=0,
) -> pd.DataFrame:
    """Computes a histogram along the index of a pd.Series time series of power readings.

    Args:
        pvt: a pd.Series or pd.DataFrame of power levels in linear units

        bounds: [lower, upper] bounds for the histogram power level bins (upper-bound inclusive)

        resolution_db: step size of the histogram power level bins

        resolution_axis: number of indices to group into a single time bin

        truncate: whether to truncate pvt to an integer factor of `resolution_axis`

        dtype: the integer data type to return for histogram counts

        axis: the axis along which to compute the histogram, if `pvt` is a DataFrame

    Returns:
        `pd.DataFrame` instance, indexed on time, columned by power transformed into dB, with values of type `dtype`

    Raises:
        ValueError: if not truncate and len(pvt) % resolution_axis != 0

    """

    if isinstance(pvt, pd.Series) and axis != 0:
        raise ValueError("axis argument is invalid for pd.Series")

    if axis == 0:
        pvt = pvt.T
    elif axis != 1:
        raise ValueError(f"axis argument must be 0 or 1")

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
