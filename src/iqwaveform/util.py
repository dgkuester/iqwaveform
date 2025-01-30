from __future__ import annotations
import importlib.util
import functools
import math
import sys
import array_api_compat
from array_api_compat import is_cupy_array
import numpy as np
from functools import lru_cache
from numbers import Number

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

    if xp == np:
        return x
    else:
        return xp.array(x, copy=False)


@lru_cache
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


def sliding_window_output_shape(array_shape: tuple|int, window_shape: tuple, axis):
    """return the shape of the output of sliding_window_view, for example
    to pre-create an output buffer."""
    window_shape = (tuple(window_shape)
                    if np.iterable(window_shape)
                    else (window_shape,))
    x_shape_trimmed = list(array_shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError(
                'window shape cannot be larger than input array shape')
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
    out_shape = sliding_window_output_shape(x.shape, window_shape, axis)
    return _cupy.lib.stride_tricks.as_strided(x, strides=out_strides, shape=out_shape)


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

    newshape = y.shape[:axis] + (ax_size // size, size) + y.shape[axis + 1 :]

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
