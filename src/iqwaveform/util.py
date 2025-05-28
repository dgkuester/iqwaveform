from __future__ import annotations
import importlib.util
import functools
import itertools
import math
from numbers import Number
import sys
import typing
import typing_extensions

import array_api_compat
from array_api_compat import is_cupy_array
import numpy as np

from contextlib import contextmanager
from enum import Enum
from . import type_stubs

__all__ = [
    'Domain',
    'set_input_domain',
    'get_input_domain',
    'NonStreamContext',
    'array_stream',
    'pad_along_axis',
    'array_namespace',
    'sliding_window_view',
    'float_dtype_like',
]

_P = typing_extensions.ParamSpec('_P')
_R = typing_extensions.TypeVar('_R')


def lazy_import(module_name: str):
    """postponed import of the module with the specified name.

    The import is not performed until the module is accessed in the code. This
    reduces the total time to import labbench by waiting to import the module
    until it is used.
    """
    # see https://docs.python.org/3/library/importlib.html#implementing-lazy-imports
    try:
        ret = sys.modules[module_name]
        return ret
    except KeyError:
        pass

    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ImportError(f'no module found named "{module_name}"')
    spec.loader = importlib.util.LazyLoader(spec.loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def binned_mean(
    x: type_stubs.ArrayType,
    count,
    *,
    axis=0,
    truncate=True,
    reject_extrema=False,
    fft=True,
) -> type_stubs.ArrayType:
    """reduce an array by averaging into bins on the specified axis.

    Arguments:
        x: input array
        count: bin count to average
        axis: axis along which to implement the binned mean
        truncate: True to truncate incomplete bins at the edges, or False to raise exception
        reject_extrema: if True, the min and max samples from each bin will be excluded
        fft: if True, bins align with fft bins (centered, instead of left side)
    """

    xp = array_namespace(x)

    if not truncate:
        pass
    elif fft:
        # enforce that index 0 is a center bin
        center_bin = x.shape[axis] // 2
        size_left = center_bin - count // 2
        blocks_left = size_left // count
        block_count = 2 * blocks_left + 1
        start = center_bin - (count * block_count) // 2
        stop = start + count * block_count

        if start > 0 or stop < x.shape[axis]:
            x = axis_slice(x, start, stop, axis=axis)
    else:
        trim = x.shape[axis] % (count)
        if trim:
            dimsize = (x.shape[axis] // count) * count
            x = axis_slice(x, 0, dimsize, axis=axis)

    x = to_blocks(x, count, axis=axis)
    stat_axis = axis + 1 if axis >= 0 else axis
    if reject_extrema:
        x = np.sort(x, axis=stat_axis)
        x = axis_slice(x, 1, -1, axis=stat_axis)
    ret = xp.nanmean(x, axis=stat_axis)
    return ret


@functools.wraps(functools.lru_cache)
def lru_cache(
    maxsize: int | None = 128, typed: bool = False
) -> typing.Callable[[typing.Callable[_P, _R]], typing.Callable[_P, _R]]:
    # presuming that the API is designed to accept only hashable types, set
    # the type hint to match the wrapped function
    return functools.lru_cache(maxsize, typed)


_input_domain = []


@lru_cache()
def find_float_inds(seq: tuple[str | float, ...]) -> list[bool]:
    """return a list to flag whether each element can be converted to float"""

    ret = []
    for s in seq:
        try:
            float(s)
        except ValueError:
            ret.append(False)
        else:
            ret.append(True)
    return ret


def isroundmod(value: float, div, atol=1e-6) -> bool:
    ratio = value / div
    try:
        return abs(math.remainder(ratio, 1)) <= atol
    except TypeError:
        return np.abs(np.rint(ratio) - ratio) <= atol


class Domain(Enum):
    TIME = 'time'
    FREQUENCY = 'frequency'
    TIME_BINNED_POWER = 'time_binned_power'


@contextmanager
def set_input_domain(domain: str | Domain):
    """set the current domain from input arrays of DSP calls"""
    i = len(_input_domain)
    _input_domain.append(Domain(domain))
    yield
    del _input_domain[i]


def get_input_domain(default=Domain.TIME):
    # validate the domain
    Domain(default)

    if len(_input_domain) > 0:
        return _input_domain[-1]
    else:
        return default


class NonStreamContext:
    """a do-nothing cupy.Stream duck type stand-in for array types that do not support synchronization"""

    def __init__(self, *args, **kws):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def synchronize(self):
        pass

    def use(self):
        pass


def array_stream(obj: type_stubs.ArrayType, null=False, non_blocking=False, ptds=False):
    """returns a cupy.Stream (or a do-nothing stand in) object as appropriate for obj"""
    if is_cupy_array(obj):
        import cupy

        return cupy.cuda.Stream(null=null, non_blocking=non_blocking, ptds=ptds)
    else:
        return NonStreamContext()


def array_namespace(a, use_compat=False):
    try:
        return array_api_compat.array_namespace(a, use_compat=use_compat)
    except TypeError:
        pass

    try:
        import mlx.core as mx

        if isinstance(a, mx.array):
            return mx
        else:
            raise TypeError
    except (ImportError, TypeError):
        pass

    raise TypeError('unrecognized object type')


def pad_along_axis(a, pad_width: list, axis=0, *args, **kws):
    if axis >= 0:
        pre_pad = [[0, 0]] * axis
    else:
        pre_pad = [[0, 0]] * (axis + a.ndim - 1)

    xp = array_namespace(a)
    return xp.pad(a, pre_pad + pad_width, *args, **kws)


@lru_cache()
def sliding_window_output_shape(array_shape: tuple | int, window_shape: tuple, axis):
    """return the shape of the output of sliding_window_view, for example
    to pre-create an output buffer."""
    try:
        # numpy >= 2?
        from numpy.lib import _stride_tricks_impl as stride_tricks
    except ImportError:
        # numpy < 2?
        from numpy.lib import stride_tricks

    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)

    if min(window_shape) < 0:
        raise ValueError('`window_shape` cannot contain negative values')

    ndim = len(array_shape)
    if axis is None:
        axis = tuple(range(ndim))
        if len(window_shape) != len(axis):
            raise ValueError(
                f'Since axis is `None`, must provide '
                f'window_shape for all dimensions of `x`; '
                f'got {len(window_shape)} window_shape elements '
                f'and `x.ndim` is {ndim}.'
            )
    else:
        axis = stride_tricks.normalize_axis_tuple(axis, ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(
                f'Must provide matching length window_shape and '
                f'axis; got {len(window_shape)} window_shape '
                f'elements and {len(axis)} axes elements.'
            )

    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)
    x_shape_trimmed = list(array_shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError('window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    return tuple(x_shape_trimmed) + window_shape


def sliding_window_view(x, window_shape, axis=None, *, subok=False, writeable=False):
    """
    Create a sliding window view into the array with the given window shape.

    Also known as rolling or moving window, the window slides across all
    dimensions of the array and extracts subsets of the array at all window
    positions.


    Parameters
    ----------
    x : array_like
        Array to create the sliding window view from.
    window_shape : int or tuple of int
        Size of window over each axis that takes part in the sliding window.
        If `axis` is not present, must have same length as the number of input
        array dimensions. Single integers `i` are treated as if they were the
        tuple `(i,)`.
    axis : int or tuple of int, optional
        Axis or axes along which the sliding window is applied.
        By default, the sliding window is applied to all axes and
        `window_shape[i]` will refer to axis `i` of `x`.
        If `axis` is given as a `tuple of int`, `window_shape[i]` will refer to
        the axis `axis[i]` of `x`.
        Single integers `i` are treated as if they were the tuple `(i,)`.
    subok : bool, optional
        If True, sub-classes will be passed-through, otherwise the returned
        array will be forced to be a base-class array (default).
    writeable : bool, optional -- not supported
        When true, allow writing to the returned view. The default is false,
        as this should be used with caution: the returned view contains the
        same memory location multiple times, so writing to one location will
        cause others to change.

    Returns
    -------
    view : ndarray
        Sliding window view of the array. The sliding window dimensions are
        inserted at the end, and the original dimensions are trimmed as
        required by the size of the sliding window.
        That is, ``view.shape = x_shape_trimmed + window_shape``, where
        ``x_shape_trimmed`` is ``x.shape`` with every entry reduced by one less
        than the corresponding window size.


    See also
    --------
    numpy.lib.stride_tricks.as_strided

    Notes
    --------
    This function is adapted from numpy.lib.stride_tricks.as_strided.

    Examples
    --------
    >>> x = _cupy.arange(6)
    >>> x.shape
    (6,)
    >>> v = sliding_window_view(x, 3)
    >>> v.shape
    (4, 3)
    >>> v
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])

    """

    try:
        # numpy >= 2?
        from numpy.lib import _stride_tricks_impl as stride_tricks
    except ImportError:
        # numpy < 2?
        from numpy.lib import stride_tricks

    xp = array_namespace(x, use_compat=False)

    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)

    # writeable is not supported:
    if writeable:
        raise NotImplementedError('Writeable views are not supported.')

    # first convert input to array, possibly keeping subclass
    x = xp.array(x, copy=False, subok=subok)

    out_shape = sliding_window_output_shape(x.shape, window_shape, axis)
    axis = stride_tricks.normalize_axis_tuple(axis, x.ndim)
    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    return xp.lib.stride_tricks.as_strided(x, strides=out_strides, shape=out_shape)


def float_dtype_like(x: type_stubs.ArrayType):
    """returns a floating-point dtype corresponding to x.

    Returns:
    * If x.dtype is float16/float32/float64: x.dtype.
    * If x.dtype is complex64/complex128: float32/float64
    """

    if isinstance(x, Number):
        x = np.asarray(x)
        xp = np
    else:
        xp = array_namespace(x)

    try:
        ret = np.finfo(xp.asarray(x).dtype).dtype
    except ValueError:
        ret = np.float32
    return ret


def to_blocks(
    y: type_stubs.ArrayType, size: int, truncate=False, axis=0
) -> type_stubs.ArrayType:
    """Returns a view on y reshaped into blocks along axis `axis`.

    Args:
        y: an input array of size (N[0], ... N[K-1])

    Raises:
        TypeError: if not isinstance(size, int)

        IndexError: if y.size == 0

        ValueError: if truncate == False and y.shape[axis] % size != 0

    Returns:
        view on `y` with shape (..., N[axis]//size, size, ..., N[K-1]])
    """

    if not isinstance(size, int):
        raise TypeError('block size must be integer')
    if y.size == 0:
        raise IndexError('cannot form blocks on arrays of size 0')

    # ensure the axis dimension is a multiple of the block size
    ax_size = y.shape[axis]
    if ax_size % size != 0:
        if not truncate:
            raise ValueError(
                f'axis 0 size {ax_size} is not a factor of block size {size}'
            )

        slices = len(y.shape) * [slice(None, None)]
        slices[axis] = slice(None, size * (ax_size // size))
        y = y.__getitem__(tuple(slices))

    if axis == -1:
        shape_after = ()
    else:
        shape_after = y.shape[axis + 1 :]
    newshape = y.shape[:axis] + (ax_size // size, size) + shape_after

    return y.reshape(newshape)


@functools.cache
def _pad_slices_to_dim(ndim: int, axis: int, /):
    if not isinstance(axis, int):
        raise TypeError('axis argument must be integer')

    if axis < 0:
        axis = ndim + axis

        if axis < 0:
            raise ValueError(f'axis {axis} exceeds the number of dimensions')

    if axis <= ndim // 2:
        before = (slice(None),) * (axis)
        after = ()
    else:
        before = (Ellipsis,)
        after = (slice(None),) * (ndim - axis - 1)

    return before, after


def axis_index(a, index, axis=-1):
    """Return a boolean-indexed selection on axis `axis' from `a'.

    Arguments:
    a: numpy.ndarray
        The array to be sliced.
    mask: boolean index array of size a.shape[axis]
    axis : int, optional
        The axis of `a` to be sliced.
    """
    before, after = _pad_slices_to_dim(a.ndim, axis)
    return a[before + (index,) + after]


def axis_slice(a, start, stop=None, step=None, axis=-1):
    """Return a slice on the array `a` on the axis index `axis`.

    Arguments:
    a: numpy.ndarray
        The array to be sliced.
    start, stop=None, step:
        The arguments to `slice` on that axis.
    axis : int, optional
        The axis of `a` to be sliced.
    """

    before, after = _pad_slices_to_dim(a.ndim, axis)
    sl = slice(start, stop, step)
    return a[before + (sl,) + after]


def histogram_last_axis(
    x: type_stubs.ArrayType, bins: int | type_stubs.ArrayType, range: tuple = None
) -> type_stubs.ArrayType:
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
    flat = x.reshape(-1, hist_size)
    idx = xp.searchsorted(bins, flat, 'right') - 1

    # Some elements would be off limits, so get a mask for those
    bad_mask = (idx == -1) | (idx == bins.size)

    # We need to use bincount to get bin based counts. To have unique IDs for
    # each row and not get confused by the ones from other rows, we need to
    # offset each row by a scale (using row length for this).
    scaled_idx = bins.size * xp.arange(flat.shape[0])[:, None] + idx

    # Set the bad ones to be last possible index+1 : n_bins*data2D.shape[0]
    limit = bins.size * flat.shape[0]
    scaled_idx[bad_mask] = limit

    # Get the counts and reshape to multi-dim
    counts = xp.bincount(scaled_idx.ravel(), minlength=limit + 1)[:-1]
    counts.shape = x.shape[:-1] + (bins.size,)
    return counts[..., :-1], bins


@lru_cache()
def dtype_change_float(dtype, float_basis_dtype) -> np.dtype:
    """return a complex or float dtype similar to `dtype`, but
    with a float backing with size matching `float_basis_dtype`.

    Examples:
        dtype_change_float(np.complex128, np.float32) -> np.complex64
        dtype_change_float(np.float64, np.float32) -> np.float32
    """

    np_input_type = np.dtype(dtype).type
    np_float_type = np.finfo(np.dtype(float_basis_dtype)).dtype.type

    if np_input_type in (np.complex128, np.complex64):
        if np_float_type is np.float32:
            return np.complex64
        elif np_float_type is np.float64:
            return np.complex128
    elif np_input_type in (np.float16, np.float32, np.float64):
        return np_float_type

    raise ValueError(
        f'unable to identify output dtype similar to {dtype} matching floating point {float_basis_dtype}'
    )


def iter_along_axes(
    x: type_stubs.ArrayType, axes: typing.Iterable[int] | None
) -> typing.Iterable[tuple[int, ...]]:
    empty_slice = slice(None, None)
    if axes is None:
        return (empty_slice,)
    elif isinstance(axes, Number):
        axes = (axes,)

    axes = [(ax if ax >= 0 else ax + x.ndim) for ax in axes]

    ax_inds = []
    for i in range(x.ndim):
        if i in axes:
            ax_inds.append(((n,) for n in range(x.shape[i])))
        else:
            ax_inds.append((empty_slice,))

    return itertools.product(*ax_inds)


def grouped_views_along_axis(x, max_size, axis=0):
    if axis < 0:
        axis = x.ndim + axis

    if x.size <= max_size:
        yield x
        return

    size_resid = max_size // x.shape[axis]

    ax_steps = []
    for iax, n in enumerate(x.shape):
        if size_resid <= 1:
            break
        if iax == axis:
            ax_steps.append((slice(None, None),))
            continue
        elif n <= size_resid:
            ax_steps.append(((i,) for i in range(n)))
        else:
            splits = list(range(0, n, size_resid))
            if splits[-1] != n - 1:
                splits.append(n)
            new = [slice(a, b) for a, b in zip(splits[:-1], splits[1:])]
            ax_steps.append(new)
        size_resid = size_resid // n

    slices = itertools.product(*ax_steps)

    empty = True
    for slice_ in slices:
        empty = False
        yield x[slice_]

    if empty:
        yield x
