from __future__ import annotations
import array_api_compat
from array_api_compat import is_cupy_array
import numpy as np
import typing


from contextlib import contextmanager
from enum import Enum

__all__ = [
    'Array',
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

if typing.TYPE_CHECKING:
    # bury this type checking in here to avoid lengthening the import time of iqwaveform
    # if cupy isn't installed
    try:
        import cupy as cp
    except ModuleNotFoundError:
        import numpy as cp

    # union of supported array types
    Array = typing.Union[np.ndarray, cp.ndarray]

else:
    Array = np.ndarray


_input_domain = []


def empty_shared(shape: tuple | int, dtype: np.dtype, xp=np):
    """allocate pinned CUDA memory that is shared between GPU/CPU on supported architectures"""

    import numba
    import numba.cuda

    x = numba.cuda.mapped_array(
        shape,
        dtype=dtype,
        strides=None,
        order='C',
        stream=0,
        portable=False,
        wc=False,
    )
    return xp.array(x, copy=False)


def isroundmod(value: float, div, atol=1e-6) -> bool:
    return np.abs(np.rint(value / div) - value / div) <= atol


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


def array_stream(obj: Array, null=False, non_blocking=False, ptds=False):
    """returns a cupy.Stream (or a do-nothing stand in) object as appropriate for obj"""
    if is_cupy_array(obj):
        import cupy

        return cupy.cuda.Stream(null=null, non_blocking=non_blocking, ptds=ptds)
    else:
        return NonStreamContext()


def array_namespace(a):
    try:
        return array_api_compat.array_namespace(a)
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

    if not is_cupy_array(x):
        return np.lib.stride_tricks.sliding_window_view(**locals())

    import cupy as _cupy

    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)

    # writeable is not supported:
    if writeable:
        raise NotImplementedError('Writeable views are not supported.')

    # first convert input to array, possibly keeping subclass
    x = _cupy.array(x, copy=False, subok=subok)

    window_shape_array = _cupy.array(window_shape)
    for dim in window_shape_array:
        if dim < 0:
            raise ValueError('`window_shape` cannot contain negative values')

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(
                f'Since axis is `None`, must provide '
                f'window_shape for all dimensions of `x`; '
                f'got {len(window_shape)} window_shape elements '
                f'and `x.ndim` is {x.ndim}.'
            )
    else:
        axis = _cupy._core.internal._normalize_axis_indices(axis, x.ndim)
        if len(window_shape) != len(axis):
            raise ValueError(
                f'Must provide matching length window_shape and '
                f'axis; got {len(window_shape)} window_shape '
                f'elements and {len(axis)} axes elements.'
            )

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError('window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return _cupy.lib.stride_tricks.as_strided(x, strides=out_strides, shape=out_shape)


def float_dtype_like(x: Array):
    """returns a floating-point dtype corresponding to x.

    Returns:
    * If x.dtype is float16/float32/float64: x.dtype.
    * If x.dtype is complex64/complex128: float32/float64
    """
    return np.finfo(x.dtype).dtype
