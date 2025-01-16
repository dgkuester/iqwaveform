"""Transformations and statistical tools for power time series"""

from .util import (
    lazy_import,
    array_namespace,
    get_input_domain,
    Domain,
    float_dtype_like,
    isroundmod,
    is_cupy_array,
    to_blocks
)

import array_api_compat.numpy as np
import functools
import re
import warnings
from numbers import Number
from functools import partial
from typing import Union, Any, Optional
from types import ModuleType
from . import type_stubs
from .type_stubs import ArrayType, ArrayLike

signal = lazy_import('scipy.signal')
pd = lazy_import('pandas')
ne = lazy_import('numexpr')
xr = lazy_import('xarray')

warnings.filterwarnings('ignore', message='.*divide by zero.*')
warnings.filterwarnings('ignore', message='.*invalid value encountered.*')


_DB_UNIT_MAPPING = {'dBm': 'mW', 'dBW': 'W', 'dB': 'unitless'}


def unit_dB_to_linear(s: str):
    for db_unit, lin_unit in _DB_UNIT_MAPPING.items():
        s, _ = re.subn('^' + db_unit, lin_unit, s, count=1)
    return s


def unit_linear_to_dB(s: str):
    for db_unit, lin_unit in _DB_UNIT_MAPPING.items():
        s, _ = re.subn('^' + lin_unit, db_unit, s, count=1)
    return s


@functools.lru_cache
def stat_ufunc_from_shorthand(kind, xp=np, axis=0):
    NAMED_UFUNCS = {
        'min': xp.min,
        'max': xp.max,
        'peak': xp.max,
        'mean': xp.mean,
        'rms': xp.mean,
    }

    if hasattr(xp, 'median'):
        NAMED_UFUNCS['median'] = xp.median

    if isinstance(kind, str):
        if kind not in NAMED_UFUNCS:
            valid = NAMED_UFUNCS.keys()
            raise ValueError(f'kind argument must be one of {valid}')
        ufunc = partial(NAMED_UFUNCS[kind], axis=axis)

    elif isinstance(kind, Number):
        ufunc = partial(xp.quantile, q=kind, axis=axis)

    elif callable(kind):
        ufunc = partial(kind, axis=axis)

    else:
        raise ValueError(f'invalid statistic ufunc "{kind}"')

    return ufunc


def _interpret_arraylike(
    x: Union[ArrayLike, Number], out: Optional[ArrayLike] = None
) -> tuple[ArrayType, ArrayType, ModuleType]:
    """interpret the array-like input and output buffer arguments.

    Returns:
        ArrayType objects pointing to the underlying array-type objects,
        and the module to work with them
    """
    try:
        xp = array_namespace(x)
        values = x
    except TypeError:
        # pandas.Series, pandas.DataFrame, xarray.DataArray
        if hasattr(x, 'values'):
            xp = array_namespace(x.values)
            values = x.values
        elif isinstance(x, Number):
            xp = np
            values = x
        else:
            raise TypeError(f'unsupported input type {type(x)}')

    if out is None:
        out = xp.zeros(xp.shape(x), dtype=float_dtype_like(values))
    elif hasattr(out, 'values'):
        # pandas, xarray objects
        out = out.values

    return values, out, xp


def _repackage_arraylike(
    values: ArrayType,
    obj: Union[ArrayLike, Number],
    *,
    unit_transform: Optional[callable] = None,
) -> Union[ArrayLike, Number]:
    """package `values` into a data type matching `obj`"""

    # accessing each of these forces imports of each module.
    # work through progressively more expensive imports
    if isinstance(obj, Number):
        return values.item()
    elif not hasattr(obj, 'values'):
        return values
    elif isinstance(obj, pd.Series):
        return pd.Series(values, index=obj.index)
    elif isinstance(obj, pd.DataFrame):
        return pd.DataFrame(values, index=obj.index, columns=obj.columns)
    elif isinstance(obj, xr.DataArray):
        ret = obj.copy(deep=False, data=values)
        units = ret.attrs.get('units', None)
        if units is not None and unit_transform is not None:
            ret.attrs = dict(units=unit_transform(units))
        return ret
    else:
        raise TypeError(f'unrecognized input type {type(obj)}')


def powtodB(
    x: Union[ArrayLike, Number], abs: bool = True, eps: float = 0, out=None
) -> Any:
    """compute `10*log10(abs(x) + eps)` or `10*log10(x + eps)` with speed optimizations"""

    eps_str = '' if eps == 0 else '+eps'

    values, out, xp = _interpret_arraylike(x, out)

    if xp is np:
        if abs:
            expr = f'real(10*log10(abs(values){eps_str}))'
        else:
            expr = f'real(10*log10(values+eps){eps_str})'
        values = ne.evaluate(expr, out=out, casting='unsafe')
    elif is_cupy_array(xp):
        from ._jit import cuda

        if eps == 0:
            if abs:
                values = cuda.powtodB(x, out)
            else:
                values = cuda.powtodB_noabs(x, out)
        else:
            if abs:
                values = cuda.powtodB_eps(x, out, eps)
            else:
                values = cuda.powtodB_eps_noabs(x, out, eps)
    else:
        # mlx, torch, etc
        # TODO: CUDA kernel evaluation here
        if abs:
            values = xp.abs(values, out=out)
        if eps != 0:
            values += eps
        values = xp.log10(values, out=out)
        values *= 10

    return _repackage_arraylike(values, x, unit_transform=unit_linear_to_dB)


def dBtopow(x: Union[ArrayLike, Number], out=None) -> Any:
    """compute `10**(x/10)` with speed optimizations"""

    values, out, xp = _interpret_arraylike(x, out)

    if xp is np:
        expr = '10**(values/10.)'
        values = ne.evaluate(expr, out=out, casting='unsafe')
    elif is_cupy_array(xp):
        from ._jit import cuda

        values = cuda.dBtopow(x, out)
    else:
        # mlx, torch, etc
        # TODO: CUDA kernel evaluation here
        values = xp.divide(values, 10, out=out)
        values = xp.power(10, values, out=out)

    return _repackage_arraylike(values, x, unit_transform=unit_dB_to_linear)


def envtopow(x: Union[ArrayLike, Number], out=None) -> Any:
    """Computes abs(x)**2 with speed optimizations"""

    values, out, xp = _interpret_arraylike(x, out)

    if xp is np:
        # numpy, pandas
        values = ne.evaluate(
            'real(abs(x)**2)', local_dict=dict(x=x), out=out, casting='unsafe'
        )

        if xp.iscomplexobj(values):
            values = values.real
    elif is_cupy_array(xp):
        from ._jit import cuda

        values = cuda.envtopow(x, out)
    else:
        # mlx, torch, etc
        # TODO: CUDA kernel evaluation here
        values = xp.abs(x, out=out)
        values *= values

    return _repackage_arraylike(values, x)


def envtodB(
    x: Union[ArrayLike, Number], abs: bool = True, eps: float = 0, out=None
) -> Any:
    """compute `20*log10(abs(x) + eps)` or `20*log10(x + eps)` with speed optimizations"""

    eps_str = '' if eps == 0 else '+eps'

    values, out, xp = _interpret_arraylike(x, out)

    if xp is np:
        if abs:
            expr = f'real(20*log10(abs(values){eps_str}))'
        else:
            expr = f'real(20*log10(values+eps){eps_str})'
        values = ne.evaluate(expr, out=out, casting='unsafe')
    elif is_cupy_array(xp):
        from ._jit import cuda

        if eps == 0:
            if abs:
                values = cuda.envtodB(x, out)
            else:
                values = cuda.envtodB_noabs(x, out)
        else:
            if abs:
                values = cuda.envtodB_eps(x, out, eps)
            else:
                values = cuda.envtodB_eps_noabs(x, out, eps)
    else:
        # mlx, torch, etc
        # TODO: CUDA kernel evaluation here
        if abs:
            values = xp.abs(values, out=out)
        if eps != 0:
            values += eps
        values = xp.log10(values, out=out)
        values *= 20

    return _repackage_arraylike(values, x, unit_transform=unit_linear_to_dB)


def iq_to_bin_power(
    iq: ArrayType,
    Ts: float,
    Tbin: float,
    randomize: bool = False,
    kind: str = 'mean',
    truncate=False,
    axis=0
):
    """computes power along the rows of `iq` (time axis) on bins of duration Tbin.

    Args:
        iq: complex-valued input waveform samples
        Ts: sample period of the input waveform
        Tbin: time duration of the bin size
        randomize: if True, randomize the start locations of the bins; otherwise, bins are contiguous
        kind: a named statistic ('max', 'mean', 'median', 'min', 'peak', 'rms'), a quantile, or a callable ufunc
        truncate: if True, truncate the last samples of `iq` to an integer number of bins
    """

    xp = array_namespace(iq)

    if truncate or isroundmod(Tbin, Ts):
        N = round(Tbin / Ts)
    else:
        raise ValueError(
            f'bin period ({Tbin} s) must be multiple of waveform sample period ({Ts})'
        )

    # instantaneous power, reshaped into bins
    if randomize:
        if axis != 0:
            raise ValueError('only axis=0 is currently supported when randomize=True')
        
        size = int(np.floor(iq.shape[0] / N))
        starts = xp.random.randint(0, iq.shape[0] - N, size)
        offsets = xp.arange(N)
        iq_blocks = iq[starts[:, np.newaxis] + offsets[np.newaxis, :]]
    else:
        iq_blocks = to_blocks(iq, N, axis=axis, truncate=truncate)

    detector = stat_ufunc_from_shorthand(kind, xp=xp, axis=axis+1)
    power_bins = envtopow(iq_blocks)

    return detector(power_bins).astype(float_dtype_like(iq))


def iq_to_cyclic_power(
    x: ArrayType,
    Ts: float,
    detector_period: float,
    cyclic_period: float,
    truncate=False,
    detectors=('rms', 'peak'),
    cycle_stats=('min', 'mean', 'max'),
    axis=0
) -> dict[str, dict[str, ArrayType]]:
    """computes a time series of periodic frame power statistics.

    The time axis on the cyclic time lag [0, cyclic_period) is binned with step size
    `detector_period`, for a total of `cyclic_period/detector_period` samples.

    RMS and peak power detector data are returned. For each type of detector, a time
    series is returned for (min, mean, max) statistics, which are computed across the
    number of frames (`cyclic_period/Ts`).

    Args:
        iq: complex-valued input waveform samples
        Ts: sample period of the iq waveform
        detector_period: sampling period within the frame
        cyclic_period: the cyclic period to analyze

    Raises:
        ValueError: if detector_period%Ts != 0 or cyclic_period%detector_period != 0

    Returns:
        dict keyed on detector type, with values (dict of np.arrays keyed on cyclic statistic)
    """

    # apply the detector statistic
    xp = array_namespace(x)
    domain = get_input_domain()

    if domain == Domain.TIME:
        # compute the binned power ourselves
        if detectors is None:
            raise ValueError(
                'supply detectors argument to evaluate binned power from time domain IQ'
            )

        power = {
            d: iq_to_bin_power(x, Ts, detector_period, kind=d, truncate=truncate, axis=axis)
            for d in detectors
        }

    elif domain == Domain.TIME_BINNED_POWER:
        # precalculated binned power
        power = x
        if not isinstance(power, dict):
            raise TypeError(
                'in time-binned power domain, expected dict input keyed by detector'
            )
        if detectors is None:
            detectors = tuple(x.keys())
        elif set(x.keys()) != set(detectors):
            raise ValueError('input data keys do not match supplied ')

    if isroundmod(cyclic_period, detector_period, atol=1e-6):
        cyclic_detector_bins = round(cyclic_period / detector_period)
    else:
        raise ValueError(
            'cyclic period must be positive integer multiple of the detector period'
        )

    power_shape = power[detectors[0]].shape
    
    if power_shape[1] % cyclic_detector_bins != 0:
        if truncate:
            N = (power_shape[1] // cyclic_detector_bins) * cyclic_detector_bins
            power = {d: x[:N] for d, x in power.items()}
        else:
            raise ValueError(
                'pass truncate=True to allow truncation to align with cyclic windows'
            )

    if axis < 0:
        axis = x.ndim + axis

    shape_by_cycle = (
        power_shape[:axis]
        + (power_shape[axis] // cyclic_detector_bins,)
        + (cyclic_detector_bins,)
        + (x.shape[axis+1:] if x.ndim > axis else ())
    )

    power = {d: x.reshape(shape_by_cycle) for d, x in power.items()}

    cycle_stat_ufunc = {
        kind: stat_ufunc_from_shorthand(kind, xp=xp) for kind in cycle_stats
    }

    # apply the cyclic statistic

    ret = {}

    for detector, x in power.items():
        ret[detector] = {}
        for cycle_stat, func in cycle_stat_ufunc.items():
            ret[detector][cycle_stat] = func(x, axis=axis)

    return ret


def iq_to_frame_power(
    iq: np.ndarray,
    Ts: float,
    detector_period: float,
    frame_period: float,
    truncate=False,
) -> dict:
    warnings.warn(
        'iq_to_frame_power has been deprecated. use iq_to_cyclic_power instead'
    )

    cyclic_period = frame_period
    del frame_period

    return iq_to_cyclic_power(**locals())


def unstack_series_to_bins(
    pvt: type_stubs.SeriesType, Tbin: float, truncate: bool = False
) -> type_stubs.DataFrameType:
    """unstack time series of power vs time (time axis) `pvt` into
    a pd.DataFrame in which row consists of time series of time duration `Twindow`.

    Arguments:

        pvt: indexed by TimedeltaIndex or TimeIndex

        Tblock: time duration of the block

    """

    Ts = pvt.index[1] - pvt.index[0]

    if not truncate and not isroundmod(Tbin, Ts):
        raise ValueError(
            'analysis window length must be multiple of the power INTEGRATION length'
        )

    N = int(np.rint(Tbin / Ts))

    pvt = pvt.iloc[: N * (pvt.shape[0] // N)]

    values = (
        pvt.values
        # insert a new axis with bin size N
        .reshape(pvt.shape[0] // N, N)
    )

    df = pd.DataFrame(values, index=pvt.index[::N], columns=pvt.index[:N])

    df.columns.name = 'Analysis window time elapsed (s)'
    df.index = pd.TimedeltaIndex(df.index, unit='s')

    return df


def sample_ccdf(a: type_stubs.ArrayType, edges: type_stubs.ArrayType, density: bool = True) -> type_stubs.ArrayType:
    """computes the fraction (or total number) of samples in `a` that
    exceed each edge value.

    Args:
        a: the vector of input samples
        edges: sample threshold values at which to characterize the distribution
        density: if True, the sample counts are normalized by `a.size`

    Returns:
        the empirical complementary cumulative distribution
    """

    xp = array_namespace(a)

    # 'left' makes the bin interval open-ended on the left side
    # (the CCDF is "number of samples exceeding interval", and not equal to)
    edge_inds = xp.searchsorted(edges, a, side='left')

    bin_counts = xp.bincount(edge_inds, minlength=edges.shape[0] + 1)
    ccdf = (a.shape[0] - bin_counts.cumsum(0))[:-1]

    if density:
        ccdf = xp.asarray(ccdf, dtype=xp.float64)
        ccdf /= a.shape[0]

    return ccdf


def hist_laxis(x: type_stubs.ArrayType, bins: int|type_stubs.ArrayType, range: tuple=None) -> type_stubs.ArrayType:
    """computes a histogram along the last axis of an input array.

    For reference see https://stackoverflow.com/questions/44152436/calculate-histograms-along-axis

    Args:
        x: input data of shape (M[0], ..., M[K-1], N)
        bins: Number of bins in the histogram, or a vector of bin edges
        range: Bounds on the histogram bins [lower bound, upper bound] inclusive

    Returns:
        np.ndarray of shape (M[0], ..., M[K-1], n_bins)
    """

    xp = array_namespace(x)

    # Setup bins and determine the bin location for each element for the bins
    hist_size = x.shape[-1]

    if isinstance(bins, int):
        if range is None:
            range = x.min(), x.max()
        bins = xp.linspace(range[0], range[1], bins + 1)
    else:
        bins = xp.asarray(bins)
    data2D = x.reshape(-1, hist_size)
    idx = xp.searchsorted(bins, data2D, 'right') - 1

    # Some elements would be off limits, so get a mask for those
    bad_mask = (idx == -1) | (idx == bins.size)

    # We need to use bincount to get bin based counts. To have unique IDs for
    # each row and not get confused by the ones from other rows, we need to
    # offset each row by a scale (using row length for this).
    scaled_idx = bins.size * xp.arange(data2D.shape[0])[:, None] + idx

    # Set the bad ones to be last possible index+1 : n_bins*data2D.shape[0]
    limit = bins.size * data2D.shape[0]
    scaled_idx[bad_mask] = limit

    # Get the counts and reshape to multi-dim
    counts = xp.bincount(scaled_idx.ravel(), minlength=limit + 1)[:-1]
    counts.shape = x.shape[:-1] + (bins.size,)
    return counts


def power_histogram_along_axis(
    pvt: type_stubs.DataFrameType,
    bounds: tuple[float, float],
    resolution_db: float,
    resolution_axis: int = 1,
    truncate: bin = True,
    dtype='uint32',
    axis=0,
) -> type_stubs.DataFrameType:
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
        raise ValueError('axis argument is invalid for pd.Series')

    if axis == 0:
        pvt = pvt.T
    elif axis != 1:
        raise ValueError('axis argument must be 0 or 1')

    # truncate to an integer number of sweep blocks
    pvt = powtodB(pvt, abs=False)

    if not truncate and len(pvt) % resolution_axis != 0:
        raise ValueError(
            'non-integer number of sweeps in pvt; pass truncate=False to truncate'
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
    power_bins = np.linspace(bounds[0], bounds[1], n_bins).astype('float64')
    df = pd.DataFrame(h, index=pvt.index[::resolution_axis], columns=power_bins)

    return df
