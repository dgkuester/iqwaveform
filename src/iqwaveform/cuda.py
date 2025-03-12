# cupy backports

import cupy as cp
from cupy.fft import rfft, fft, fftfreq, ifftshift, irfft, ifft, get_window
from math import gcd
from cupyx.scipy.signal import firwin

"""
upfirdn implementation.

Functions defined here were ported directly from cuSignal under
terms of the MIT license, under the following notice:

Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

"""

from math import ceil
import cupy

_upfirdn_modes = [
    'constant',
    'wrap',
    'edge',
    'smooth',
    'symmetric',
    'reflect',
    'antisymmetric',
    'antireflect',
    'line',
]


UPFIRDN_KERNEL = r"""
#include <cupy/complex.cuh>

///////////////////////////////////////////////////////////////////////////////
//                              UPFIRDN1D                                    //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void _cupy_upfirdn1D( const T *__restrict__ inp,
                                 const T *__restrict__ h_trans_flip,
                                 const int up,
                                 const int down,
                                 const int axis,
                                 const int x_shape_a,
                                 const int h_per_phase,
                                 const int padded_len,
                                 T *__restrict__ out,
                                 const int outW ) {

    const int t { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    for ( size_t tid = t; tid < outW; tid += stride ) {

#if ( __CUDACC_VER_MAJOR__ >= 11 ) && ( __CUDACC_VER_MINOR__ >= 2 )
        __builtin_assume( padded_len > 0 );
        __builtin_assume( up > 0 );
        __builtin_assume( down > 0 );
        __builtin_assume( tid > 0 );
#endif

        const int x_idx { static_cast<int>( ( tid * down ) / up ) % padded_len };
        int       h_idx { static_cast<int>( ( tid * down ) % up * h_per_phase ) };
        int       x_conv_idx { x_idx - h_per_phase + 1 };

        if ( x_conv_idx < 0 ) {
            h_idx -= x_conv_idx;
            x_conv_idx = 0;
        }

        T temp {};

        int stop = ( x_shape_a < ( x_idx + 1 ) ) ? x_shape_a : ( x_idx + 1 );

        for ( int x_c = x_conv_idx; x_c < stop; x_c++ ) {
            temp += inp[x_c] * h_trans_flip[h_idx];
            h_idx += 1;
        }
        out[tid] = temp;
    }
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_upfirdn1D_float32( const float *__restrict__ inp,
                                                                             const float *__restrict__ h_trans_flip,
                                                                             const int up,
                                                                             const int down,
                                                                             const int axis,
                                                                             const int x_shape_a,
                                                                             const int h_per_phase,
                                                                             const int padded_len,
                                                                             float *__restrict__ out,
                                                                             const int outW ) {
    _cupy_upfirdn1D<float>( inp, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_upfirdn1D_float64( const double *__restrict__ inp,
                                                                             const double *__restrict__ h_trans_flip,
                                                                             const int up,
                                                                             const int down,
                                                                             const int axis,
                                                                             const int x_shape_a,
                                                                             const int h_per_phase,
                                                                             const int padded_len,
                                                                             double *__restrict__ out,
                                                                             const int outW ) {
    _cupy_upfirdn1D<double>( inp, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_upfirdn1D_complex64( const thrust::complex<float> *__restrict__ inp,
                               const thrust::complex<float> *__restrict__ h_trans_flip,
                               const int up,
                               const int down,
                               const int axis,
                               const int x_shape_a,
                               const int h_per_phase,
                               const int padded_len,
                               thrust::complex<float> *__restrict__ out,
                               const int outW ) {
    _cupy_upfirdn1D<thrust::complex<float>>(
        inp, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_upfirdn1D_complex128( const thrust::complex<double> *__restrict__ inp,
                                const thrust::complex<double> *__restrict__ h_trans_flip,
                                const int up,
                                const int down,
                                const int axis,
                                const int x_shape_a,
                                const int h_per_phase,
                                const int padded_len,
                                thrust::complex<double> *__restrict__ out,
                                const int outW ) {
    _cupy_upfirdn1D<thrust::complex<double>>(
        inp, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW );
}

///////////////////////////////////////////////////////////////////////////////
//                              UPFIRDN2D                                    //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void _cupy_upfirdn2D( const T *__restrict__ inp,
                                 const int inpH,
                                 const T *__restrict__ h_trans_flip,
                                 const int up,
                                 const int down,
                                 const int axis,
                                 const int x_shape_a,
                                 const int h_per_phase,
                                 const int padded_len,
                                 T *__restrict__ out,
                                 const int outW,
                                 const int outH ) {

    const int ty { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int tx { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };

    const int stride_y { static_cast<int>( blockDim.x * gridDim.x ) };
    const int stride_x { static_cast<int>( blockDim.y * gridDim.y ) };

    for ( int x = tx; x < outH; x += stride_x ) {
        for ( int y = ty; y < outW; y += stride_y ) {
            int x_idx {};
            int h_idx {};

#if ( __CUDACC_VER_MAJOR__ >= 11 ) && ( __CUDACC_VER_MINOR__ >= 2 )
            __builtin_assume( padded_len > 0 );
            __builtin_assume( up > 0 );
            __builtin_assume( down > 0 );
#endif

            if ( axis == 1 ) {
#if ( __CUDACC_VER_MAJOR__ >= 11 ) && ( __CUDACC_VER_MINOR__ >= 2 )
                __builtin_assume( x > 0 );
#endif
                x_idx = ( static_cast<int>( x * down ) / up ) % padded_len;
                h_idx = ( x * down ) % up * h_per_phase;
            } else {
#if ( __CUDACC_VER_MAJOR__ >= 11 ) && ( __CUDACC_VER_MINOR__ >= 2 )
                __builtin_assume( y > 0 );
#endif
                x_idx = ( static_cast<int>( y * down ) / up ) % padded_len;
                h_idx = ( y * down ) % up * h_per_phase;
            }

            int x_conv_idx { x_idx - h_per_phase + 1 };
            if ( x_conv_idx < 0 ) {
                h_idx -= x_conv_idx;
                x_conv_idx = 0;
            }

            T temp {};

            int stop = ( x_shape_a < ( x_idx + 1 ) ) ? x_shape_a : ( x_idx + 1 );

            for ( int x_c = x_conv_idx; x_c < stop; x_c++ ) {
                if ( axis == 1 ) {
                    temp += inp[y * inpH + x_c] * h_trans_flip[h_idx];
                } else {
                    temp += inp[x_c * inpH + x] * h_trans_flip[h_idx];
                }
                h_idx += 1;
            }
            out[y * outH + x] = temp;
        }
    }
}

extern "C" __global__ void __launch_bounds__( 64 ) _cupy_upfirdn2D_float32( const float *__restrict__ inp,
                                                                            const int inpH,
                                                                            const float *__restrict__ h_trans_flip,
                                                                            const int up,
                                                                            const int down,
                                                                            const int axis,
                                                                            const int x_shape_a,
                                                                            const int h_per_phase,
                                                                            const int padded_len,
                                                                            float *__restrict__ out,
                                                                            const int outW,
                                                                            const int outH ) {
    _cupy_upfirdn2D<float>(
        inp, inpH, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW, outH );
}

extern "C" __global__ void _cupy_upfirdn2D_float64( const double *__restrict__ inp,
                                                    const int inpH,
                                                    const double *__restrict__ h_trans_flip,
                                                    const int up,
                                                    const int down,
                                                    const int axis,
                                                    const int x_shape_a,
                                                    const int h_per_phase,
                                                    const int padded_len,
                                                    double *__restrict__ out,
                                                    const int outW,
                                                    const int outH ) {
    _cupy_upfirdn2D<double>(
        inp, inpH, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW, outH );
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_upfirdn2D_complex64( const thrust::complex<float> *__restrict__ inp,
                               const int inpH,
                               const thrust::complex<float> *__restrict__ h_trans_flip,
                               const int up,
                               const int down,
                               const int axis,
                               const int x_shape_a,
                               const int h_per_phase,
                               const int padded_len,
                               thrust::complex<float> *__restrict__ out,
                               const int outW,
                               const int outH ) {
    _cupy_upfirdn2D<thrust::complex<float>>(
        inp, inpH, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW, outH );
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_upfirdn2D_complex128( const thrust::complex<double> *__restrict__ inp,
                                const int inpH,
                                const thrust::complex<double> *__restrict__ h_trans_flip,
                                const int up,
                                const int down,
                                const int axis,
                                const int x_shape_a,
                                const int h_per_phase,
                                const int padded_len,
                                thrust::complex<double> *__restrict__ out,
                                const int outW,
                                const int outH ) {
    _cupy_upfirdn2D<thrust::complex<double>>(
        inp, inpH, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW, outH );
}
"""  # NOQA


UPFIRDN_MODULE = cupy.RawModule(
    code=UPFIRDN_KERNEL,
    options=('-std=c++11',),
    name_expressions=[
        '_cupy_upfirdn1D_float32',
        '_cupy_upfirdn1D_float64',
        '_cupy_upfirdn1D_complex64',
        '_cupy_upfirdn1D_complex128',
        '_cupy_upfirdn2D_float32',
        '_cupy_upfirdn2D_float64',
        '_cupy_upfirdn2D_complex64',
        '_cupy_upfirdn2D_complex128',
    ],
)


def _pad_h(h, up):
    """Store coefficients in a transposed, flipped arrangement.
    For example, suppose upRate is 3, and the
    input number of coefficients is 10, represented as h[0], ..., h[9].
    Then the internal buffer will look like this::
       h[9], h[6], h[3], h[0],   // flipped phase 0 coefs
       0,    h[7], h[4], h[1],   // flipped phase 1 coefs (zero-padded)
       0,    h[8], h[5], h[2],   // flipped phase 2 coefs (zero-padded)
    """
    h_padlen = len(h) + (-len(h) % up)
    h_full = cupy.zeros(h_padlen, h.dtype)
    h_full[: len(h)] = h
    h_full = h_full.reshape(-1, up).T[:, ::-1].ravel()
    return h_full


def _output_len(len_h, in_len, up, down):
    return (((in_len - 1) * up + len_h) - 1) // down + 1


# These three _get_* functions are vendored from
# https://github.com/rapidsai/cusignal/blob/branch-23.08/python/cusignal/utils/helper_tools.py#L55
def _get_max_gdx():
    device_id = cupy.cuda.Device()
    return device_id.attributes['MaxGridDimX']


def _get_max_gdy():
    device_id = cupy.cuda.Device()
    return device_id.attributes['MaxGridDimY']


def _get_tpb_bpg():
    device_id = cupy.cuda.Device()
    numSM = device_id.attributes['MultiProcessorCount']
    threadsperblock = 512
    blockspergrid = numSM * 20

    return threadsperblock, blockspergrid


class _UpFIRDn(object):
    def __init__(self, h, x_dtype, up, down):
        """Helper for resampling"""
        h = cupy.asarray(h)
        if h.ndim != 1 or h.size == 0:
            raise ValueError('h must be 1D with non-zero length')

        self._output_type = cupy.result_type(h.dtype, x_dtype, cupy.float32)
        h = cupy.asarray(h, self._output_type)
        self._up = int(up)
        self._down = int(down)
        if self._up < 1 or self._down < 1:
            raise ValueError('Both up and down must be >= 1')
        # This both transposes, and "flips" each phase for filtering
        self._h_trans_flip = _pad_h(h, self._up)
        self._h_trans_flip = cupy.asarray(self._h_trans_flip)
        self._h_trans_flip = cupy.ascontiguousarray(self._h_trans_flip)
        self._h_len_orig = len(h)

    def apply_filter(self, x, axis, out=None):
        """Apply the prepared filter to the specified axis of a nD signal x"""

        x = cupy.asarray(x, self._output_type)

        output_len = _output_len(self._h_len_orig, x.shape[axis], self._up, self._down)
        output_shape = list(x.shape)
        output_shape[axis] = output_len
        if out is None:
            out = cupy.empty(output_shape, dtype=self._output_type, order='C')
        axis = axis % x.ndim

        # Precompute variables on CPU
        x_shape_a = x.shape[axis]
        h_per_phase = len(self._h_trans_flip) // self._up
        padded_len = x.shape[axis] + (len(self._h_trans_flip) // self._up) - 1

        if out.ndim == 1:
            threadsperblock, blockspergrid = _get_tpb_bpg()

            kernel = UPFIRDN_MODULE.get_function(f'_cupy_upfirdn1D_{out.dtype.name}')
            kernel(
                ((x.shape[0] + 128 - 1) // 128,),
                (128,),
                (
                    x,
                    self._h_trans_flip,
                    self._up,
                    self._down,
                    axis,
                    x_shape_a,
                    h_per_phase,
                    padded_len,
                    out,
                    out.shape[0],
                ),
            )

        elif out.ndim == 2:
            # set up the kernel launch parameters
            threadsperblock = (8, 8)
            blocks = ceil(out.shape[0] / threadsperblock[0])
            blockspergrid_x = blocks if blocks < _get_max_gdx() else _get_max_gdx()

            blocks = ceil(out.shape[1] / threadsperblock[1])
            blockspergrid_y = blocks if blocks < _get_max_gdy() else _get_max_gdy()

            blockspergrid = (blockspergrid_x, blockspergrid_y)

            # do computations
            kernel = UPFIRDN_MODULE.get_function(f'_cupy_upfirdn2D_{out.dtype.name}')
            kernel(
                threadsperblock,
                blockspergrid,
                (
                    x,
                    x.shape[1],
                    self._h_trans_flip,
                    self._up,
                    self._down,
                    axis,
                    x_shape_a,
                    h_per_phase,
                    padded_len,
                    out,
                    out.shape[0],
                    out.shape[1],
                ),
            )
        else:
            raise NotImplementedError('upfirdn() requires ndim <= 2')

        return out


def upfirdn(h, x, up=1, down=1, axis=-1, mode='constant', cval=0, out=None):
    """
    Upsample, FIR filter, and downsample.

    Parameters
    ----------
    h : array_like
        1-dimensional FIR (finite-impulse response) filter coefficients.
    x : array_like
        Input signal array.
    up : int, optional
        Upsampling rate. Default is 1.
    down : int, optional
        Downsampling rate. Default is 1.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis. Default is -1.
    mode : str, optional
        This parameter is not implemented for values other than ``"constant"``.
    cval : float, optional
        This parameter is not implemented for values other than 0.

    Returns
    -------
    y : ndarray
        The output signal array. Dimensions will be the same as `x` except
        for along `axis`, which will change size according to the `h`,
        `up`,  and `down` parameters.

    Notes
    -----
    The algorithm is an implementation of the block diagram shown on page 129
    of the Vaidyanathan text [1]_ (Figure 4.3-8d).

    The direct approach of upsampling by factor of P with zero insertion,
    FIR filtering of length ``N``, and downsampling by factor of Q is
    O(N*Q) per output sample. The polyphase implementation used here is
    O(N/P).

    See Also
    --------
    scipy.signal.upfirdn

    References
    ----------
    .. [1] P. P. Vaidyanathan, Multirate Systems and Filter Banks,
       Prentice Hall, 1993.
    """
    if mode is None:
        mode = 'constant'  # For backwards compatibility
    if mode != 'constant' or cval != 0:
        raise NotImplementedError(f'{mode = } and {cval =} not implemented.')

    ufd = _UpFIRDn(h, x.dtype, int(up), int(down))
    # This is equivalent to (but faster than) using cp.apply_along_axis
    return ufd.apply_filter(x, axis, out=out)


def resample(x, num, t=None, axis=0, window=None, domain="time"):
    """
    Resample `x` to `num` samples using Fourier method along the given axis.

    The resampled signal starts at the same value as `x` but is sampled
    with a spacing of ``len(x) / num * (spacing of x)``.  Because a
    Fourier method is used, the signal is assumed to be periodic.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    num : int
        The number of samples in the resampled signal.
    t : array_like, optional
        If `t` is given, it is assumed to be the sample positions
        associated with the signal data in `x`.
    axis : int, optional
        The axis of `x` that is resampled.  Default is 0.
    window : array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier
        domain.  See below for details.
    domain : string, optional
        A string indicating the domain of the input `x`:

        ``time``
           Consider the input `x` as time-domain. (Default)
        ``freq``
           Consider the input `x` as frequency-domain.

    Returns
    -------
    resampled_x or (resampled_x, resampled_t)
        Either the resampled array, or, if `t` was given, a tuple
        containing the resampled array and the corresponding resampled
        positions.

    See Also
    --------
    decimate : Downsample the signal after applying an FIR or IIR filter.
    resample_poly : Resample using polyphase filtering and an FIR filter.

    Notes
    -----
    The argument `window` controls a Fourier-domain window that tapers
    the Fourier spectrum before zero-padding to alleviate ringing in
    the resampled values for sampled signals you didn't intend to be
    interpreted as band-limited.

    If `window` is a function, then it is called with a vector of inputs
    indicating the frequency bins (i.e. fftfreq(x.shape[axis]) ).

    If `window` is an array of the same length as `x.shape[axis]` it is
    assumed to be the window to be applied directly in the Fourier
    domain (with dc and low-frequency first).

    For any other type of `window`, the function `cusignal.get_window`
    is called to generate the window.

    The first sample of the returned vector is the same as the first
    sample of the input vector.  The spacing between samples is changed
    from ``dx`` to ``dx * len(x) / num``.

    If `t` is not None, then it represents the old sample positions,
    and the new sample positions will be returned as well as the new
    samples.

    As noted, `resample` uses FFT transformations, which can be very
    slow if the number of input or output samples is large and prime;
    see `scipy.fftpack.fft`.

    Examples
    --------
    Note that the end of the resampled data rises to meet the first
    sample of the next cycle:

    >>> import cupy as cp
    >>> import cupyx.scipy.signal import resample

    >>> x = cupy.linspace(0, 10, 20, endpoint=False)
    >>> y = cupy.cos(-x**2/6.0)
    >>> f = resample(y, 100)
    >>> xnew = cupy.linspace(0, 10, 100, endpoint=False)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(cupy.asnumpy(x), cupy.asnumpy(y), 'go-', cupy.asnumpy(xnew), \
                cupy.asnumpy(f), '.-', 10, cupy.asnumpy(y[0]), 'ro')
    >>> plt.legend(['data', 'resampled'], loc='best')
    >>> plt.show()
    """
    if domain not in ('time', 'freq'):
        raise ValueError("Acceptable domain flags are 'time' or"
                         " 'freq', not domain={}".format(domain))

    x = cupy.asarray(x)
    Nx = x.shape[axis]

    # Check if we can use faster real FFT
    real_input = cupy.isrealobj(x)

    if domain == 'time':
        # Forward transform
        if real_input:
            X = rfft(x, axis=axis)
        else:  # Full complex FFT
            X = fft(x, axis=axis)
    else:  # domain == 'freq'
        X = x

    # Apply window to spectrum
    if window is not None:
        if callable(window):
            W = window(fftfreq(Nx))
        elif isinstance(window, cupy.ndarray):
            if window.shape != (Nx,):
                raise ValueError('window must have the same length as data')
            W = window
        else:
            W = ifftshift(get_window(window, Nx))

        newshape_W = [1] * x.ndim
        newshape_W[axis] = X.shape[axis]
        if real_input:
            # Fold the window back on itself to mimic complex behavior
            W_real = W.copy()
            W_real[1:] += W_real[-1:0:-1]
            W_real[1:] *= 0.5
            X *= W_real[:newshape_W[axis]].reshape(newshape_W)
        else:
            X *= W.reshape(newshape_W)

    # Copy each half of the original spectrum to the output spectrum, either
    # truncating high frequencies (downsampling) or zero-padding them
    # (upsampling)

    # Placeholder array for output spectrum
    newshape = list(x.shape)
    if real_input:
        newshape[axis] = num // 2 + 1
    else:
        newshape[axis] = num
    Y = cupy.zeros(newshape, X.dtype)

    # Copy positive frequency components (and Nyquist, if present)
    N = min(num, Nx)
    nyq = N // 2 + 1  # Slice index that includes Nyquist if present
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, nyq)
    Y[tuple(sl)] = X[tuple(sl)]
    if not real_input:
        # Copy negative frequency components
        if N > 2:  # (slice expression doesn't collapse to empty array)
            sl[axis] = slice(nyq - N, None)
            Y[tuple(sl)] = X[tuple(sl)]

    # Split/join Nyquist component(s) if present
    # So far we have set Y[+N/2]=X[+N/2]
    if N % 2 == 0:
        if num < Nx:  # downsampling
            if real_input:
                sl[axis] = slice(N//2, N//2 + 1)
                Y[tuple(sl)] *= 2.
            else:
                # select the component of Y at frequency +N/2,
                # add the component of X at -N/2
                sl[axis] = slice(-N//2, -N//2 + 1)
                Y[tuple(sl)] += X[tuple(sl)]
        elif Nx < num:  # upsampling
            # select the component at frequency +N/2 and halve it
            sl[axis] = slice(N//2, N//2 + 1)
            Y[tuple(sl)] *= 0.5
            if not real_input:
                temp = Y[tuple(sl)]
                # set the component at -N/2 equal to the component at +N/2
                sl[axis] = slice(num-N//2, num-N//2 + 1)
                Y[tuple(sl)] = temp

    # Inverse transform
    if real_input:
        y = irfft(Y, num, axis=axis)
    else:
        y = ifft(Y, axis=axis, overwrite_x=True)

    y *= (float(num) / float(Nx))

    if t is None:
        return y
    else:
        new_t = cupy.arange(0, num) * (t[1] - t[0]) * Nx / float(num) + t[0]
        return y, new_t


def resample_poly(x, up, down, axis=0, window=("kaiser", 5.0),
                  padtype='constant', cval=None):
    """
    Resample `x` along the given axis using polyphase filtering.

    The signal `x` is upsampled by the factor `up`, a zero-phase low-pass
    FIR filter is applied, and then it is downsampled by the factor `down`.
    The resulting sample rate is ``up / down`` times the original sample
    rate. Values beyond the boundary of the signal are assumed to be zero
    during the filtering step.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.
    axis : int, optional
        The axis of `x` that is resampled. Default is 0.
    window : string, tuple, or array_like, optional
        Desired window to use to design the low-pass filter, or the FIR filter
        coefficients to employ. See below for details.
    padtype : string, optional
        `constant`, `line`, `mean`, `median`, `maximum`, `minimum` or any of
        the other signal extension modes supported by
        `cupyx.scipy.signal.upfirdn`. Changes assumptions on values beyond
        the boundary. If `constant`, assumed to be `cval` (default zero).
        If `line` assumed to continue a linear trend defined by the first and
        last points. `mean`, `median`, `maximum` and `minimum` work as in
        `cupy.pad` and assume that the values beyond the boundary are the mean,
        median, maximum or minimum respectively of the array along the axis.
    cval : float, optional
        Value to use if `padtype='constant'`. Default is zero.

    Returns
    -------
    resampled_x : array
        The resampled array.

    See Also
    --------
    decimate : Downsample the signal after applying an FIR or IIR filter.
    resample : Resample up or down using the FFT method.

    Notes
    -----
    This polyphase method will likely be faster than the Fourier method
    in `cusignal.resample` when the number of samples is large and
    prime, or when the number of samples is large and `up` and `down`
    share a large greatest common denominator. The length of the FIR
    filter used will depend on ``max(up, down) // gcd(up, down)``, and
    the number of operations during polyphase filtering will depend on
    the filter length and `down` (see `cusignal.upfirdn` for details).

    The argument `window` specifies the FIR low-pass filter design.

    If `window` is an array_like it is assumed to be the FIR filter
    coefficients. Note that the FIR filter is applied after the upsampling
    step, so it should be designed to operate on a signal at a sampling
    frequency higher than the original by a factor of `up//gcd(up, down)`.
    This function's output will be centered with respect to this array, so it
    is best to pass a symmetric filter with an odd number of samples if, as
    is usually the case, a zero-phase filter is desired.

    For any other type of `window`, the functions `cusignal.get_window`
    and `cusignal.firwin` are called to generate the appropriate filter
    coefficients.

    The first sample of the returned vector is the same as the first
    sample of the input vector. The spacing between samples is changed
    from ``dx`` to ``dx * down / float(up)``.

    Examples
    --------
    Note that the end of the resampled data rises to meet the first
    sample of the next cycle for the FFT method, and gets closer to zero
    for the polyphase method:

    >>> import cupy
    >>> import cupyx.scipy.signal import resample, resample_poly

    >>> x = cupy.linspace(0, 10, 20, endpoint=False)
    >>> y = cupy.cos(-x**2/6.0)
    >>> f_fft = resample(y, 100)
    >>> f_poly = resample_poly(y, 100, 20)
    >>> xnew = cupy.linspace(0, 10, 100, endpoint=False)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(cupy.asnumpy(xnew), cupy.asnumpy(f_fft), 'b.-', \
                 cupy.asnumpy(xnew), cupy.asnumpy(f_poly), 'r.-')
    >>> plt.plot(cupy.asnumpy(x), cupy.asnumpy(y), 'ko-')
    >>> plt.plot(10, cupy.asnumpy(y[0]), 'bo', 10, 0., 'ro')  # boundaries
    >>> plt.legend(['resample', 'resamp_poly', 'data'], loc='best')
    >>> plt.show()
    """

    if padtype != 'constant' or cval is not None:
        raise ValueError(
            'padtype and cval arguments are not supported by upfirdn')

    x = cupy.asarray(x)
    up = int(up)
    down = int(down)
    if up < 1 or down < 1:
        raise ValueError("up and down must be >= 1")

    # Determine our up and down factors
    # Use a rational approximation to save computation time on really long
    # signals
    g_ = gcd(up, down)
    up //= g_
    down //= g_
    if up == down == 1:
        return x.copy()
    n_out = x.shape[axis] * up
    n_out = n_out // down + bool(n_out % down)

    if isinstance(window, (list, cupy.ndarray)):
        window = cupy.asarray(window)
        if window.ndim > 1:
            raise ValueError("window must be 1-D")
        half_len = (window.size - 1) // 2
        h = up * window
    else:
        half_len = 10 * max(up, down)
        h = up * _design_resample_poly(up, down, window)

    # Zero-pad our filter to put the output samples at the center
    n_pre_pad = down - half_len % down
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down
    # We should rarely need to do this given our filter lengths...
    while (
        _output_len(len(h) + n_pre_pad + n_post_pad, x.shape[axis], up, down)
        < n_out + n_pre_remove
    ):
        n_post_pad += 1

    h = cupy.concatenate(
        (cupy.zeros(n_pre_pad, h.dtype), h, cupy.zeros(n_post_pad, h.dtype)))
    n_pre_remove_end = n_pre_remove + n_out

    # filter then remove excess
    y = upfirdn(h, x, up, down, axis)
    keep = [slice(None)] * x.ndim
    keep[axis] = slice(n_pre_remove, n_pre_remove_end)

    return y[tuple(keep)]


def _design_resample_poly(up, down, window):
    """
    Design a prototype FIR low-pass filter using the window method
    for use in polyphase rational resampling.

    Parameters
    ----------
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.
    window : string or tuple
        Desired window to use to design the low-pass filter.
        See below for details.

    Returns
    -------
    h : array
        The computed FIR filter coefficients.

    See Also
    --------
    resample_poly : Resample up or down using the polyphase method.

    Notes
    -----
    The argument `window` specifies the FIR low-pass filter design.
    The functions `cusignal.get_window` and `cusignal.firwin`
    are called to generate the appropriate filter coefficients.

    The returned array of coefficients will always be of data type
    `complex128` to maintain precision. For use in lower-precision
    filter operations, this array should be converted to the desired
    data type before providing it to `cusignal.resample_poly`.

    """

    # Determine our up and down factors
    # Use a rational approximation to save computation time on really long
    # signals
    g_ = gcd(up, down)
    up //= g_
    down //= g_

    # Design a linear-phase low-pass FIR filter
    max_rate = max(up, down)
    f_c = 1.0 / max_rate  # cutoff of FIR filter (rel. to Nyquist)

    # reasonable cutoff for our sinc-like function
    half_len = 10 * max_rate

    h = firwin(2 * half_len + 1, f_c, window=window)
    return h


# Scipy <= 1.12 has a deprecated `nyq` argument (nyq = fs/2).
# Remove it here, to be forward-looking.
def firwin(
    numtaps,
    cutoff,
    width=None,
    window="hamming",
    pass_zero=True,
    scale=True,
    fs=2,
):
    """
    FIR filter design using the window method.

    This function computes the coefficients of a finite impulse response
    filter.  The filter will have linear phase; it will be Type I if
    `numtaps` is odd and Type II if `numtaps` is even.

    Type II filters always have zero response at the Nyquist frequency, so a
    ValueError exception is raised if firwin is called with `numtaps` even and
    having a passband whose right end is at the Nyquist frequency.

    Parameters
    ----------
    numtaps : int
        Length of the filter (number of coefficients, i.e. the filter
        order + 1).  `numtaps` must be odd if a passband includes the
        Nyquist frequency.
    cutoff : float or 1D array_like
        Cutoff frequency of filter (expressed in the same units as `fs`)
        OR an array of cutoff frequencies (that is, band edges). In the
        latter case, the frequencies in `cutoff` should be positive and
        monotonically increasing between 0 and `fs/2`.  The values 0 and
        `fs/2` must not be included in `cutoff`.
    width : float or None, optional
        If `width` is not None, then assume it is the approximate width
        of the transition region (expressed in the same units as `fs`)
        for use in Kaiser FIR filter design.  In this case, the `window`
        argument is ignored.
    window : string or tuple of string and parameter values, optional
        Desired window to use. See `cusignal.get_window` for a list
        of windows and required parameters.
    pass_zero : {True, False, 'bandpass', 'lowpass', 'highpass', 'bandstop'},
        optional
        If True, the gain at the frequency 0 (i.e. the "DC gain") is 1.
        If False, the DC gain is 0. Can also be a string argument for the
        desired filter type (equivalent to ``btype`` in IIR design functions).
    scale : bool, optional
        Set to True to scale the coefficients so that the frequency
        response is exactly unity at a certain frequency.
        That frequency is either:

        - 0 (DC) if the first passband starts at 0 (i.e. pass_zero
          is True)
        - `fs/2` (the Nyquist frequency) if the first passband ends at
          `fs/2` (i.e the filter is a single band highpass filter);
          center of first passband otherwise
    fs : float, optional
        The sampling frequency of the signal.  Each frequency in `cutoff`
        must be between 0 and ``fs/2``.  Default is 2.

    Returns
    -------
    h : (numtaps,) ndarray
        Coefficients of length `numtaps` FIR filter.

    Raises
    ------
    ValueError
        If any value in `cutoff` is less than or equal to 0 or greater
        than or equal to ``fs/2``, if the values in `cutoff` are not strictly
        monotonically increasing, or if `numtaps` is even but a passband
        includes the Nyquist frequency.

    See Also
    --------
    firwin2
    firls
    minimum_phase
    remez

    Examples
    --------
    Low-pass from 0 to f:

    >>> import cusignal
    >>> numtaps = 3
    >>> f = 0.1
    >>> cusignal.firwin(numtaps, f)
    array([ 0.06799017,  0.86401967,  0.06799017])

    Use a specific window function:

    >>> cusignal.firwin(numtaps, f, window='nuttall')
    array([  3.56607041e-04,   9.99286786e-01,   3.56607041e-04])

    High-pass ('stop' from 0 to f):

    >>> cusignal.firwin(numtaps, f, pass_zero=False)
    array([-0.00859313,  0.98281375, -0.00859313])

    Band-pass:

    >>> f1, f2 = 0.1, 0.2
    >>> cusignal.firwin(numtaps, [f1, f2], pass_zero=False)
    array([ 0.06301614,  0.88770441,  0.06301614])

    Band-stop:

    >>> cusignal.firwin(numtaps, [f1, f2])
    array([-0.00801395,  1.0160279 , -0.00801395])

    Multi-band (passbands are [0, f1], [f2, f3] and [f4, 1]):

    >>> f3, f4 = 0.3, 0.4
    >>> cusignal.firwin(numtaps, [f1, f2, f3, f4])
    array([-0.01376344,  1.02752689, -0.01376344])

    Multi-band (passbands are [f1, f2] and [f3,f4]):

    >>> cusignal.firwin(numtaps, [f1, f2, f3, f4], pass_zero=False)
    array([ 0.04890915,  0.91284326,  0.04890915])

    """

    nyq = 0.5 * fs

    cutoff = cupy.atleast_1d(cutoff) / float(nyq)

    # Check for invalid input.
    if cutoff.ndim > 1:
        raise ValueError(
            "The cutoff argument must be at most " "one-dimensional.")
    if cutoff.size == 0:
        raise ValueError("At least one cutoff frequency must be given.")
    if cutoff.min() <= 0 or cutoff.max() >= 1:
        raise ValueError(
            "Invalid cutoff frequency: frequencies must be "
            "greater than 0 and less than nyq."
        )
    if cupy.any(cupy.diff(cutoff) <= 0):
        raise ValueError(
            "Invalid cutoff frequencies: the frequencies "
            "must be strictly increasing."
        )

    if width is not None:
        # A width was given.  Find the beta parameter of the Kaiser window
        # and set `window`.  This overrides the value of `window` passed in.
        atten = kaiser_atten(numtaps, float(width) / nyq)
        beta = kaiser_beta(atten)
        window = ("kaiser", beta)

    if isinstance(pass_zero, str):
        if pass_zero in ("bandstop", "lowpass"):
            if pass_zero == "lowpass":
                if cutoff.size != 1:
                    raise ValueError(
                        "cutoff must have one element if "
                        'pass_zero=="lowpass", got %s' % (cutoff.shape,)
                    )
            elif cutoff.size <= 1:
                raise ValueError(
                    "cutoff must have at least two elements if "
                    'pass_zero=="bandstop", got %s' % (cutoff.shape,)
                )
            pass_zero = True
        elif pass_zero in ("bandpass", "highpass"):
            if pass_zero == "highpass":
                if cutoff.size != 1:
                    raise ValueError(
                        "cutoff must have one element if "
                        'pass_zero=="highpass", got %s' % (cutoff.shape,)
                    )
            elif cutoff.size <= 1:
                raise ValueError(
                    "cutoff must have at least two elements if "
                    'pass_zero=="bandpass", got %s' % (cutoff.shape,)
                )
            pass_zero = False
        else:
            raise ValueError(
                'pass_zero must be True, False, "bandpass", '
                '"lowpass", "highpass", or "bandstop", got '
                "{}".format(pass_zero)
            )

    pass_nyquist = bool(cutoff.size & 1) ^ pass_zero

    if pass_nyquist and numtaps % 2 == 0:
        raise ValueError(
            "A filter with an even number of coefficients must "
            "have zero response at the Nyquist rate."
        )

    # Insert 0 and/or 1 at the ends of cutoff so that the length of cutoff
    # is even, and each pair in cutoff corresponds to passband.
    cutoff = cupy.hstack(([0.0] * pass_zero, cutoff, [1.0] * pass_nyquist))

    # `bands` is a 2D array; each row gives the left and right edges of
    # a passband.
    bands = cutoff.reshape(-1, 2)

    win = get_window(window, numtaps, fftbins=False)
    h, hc = _firwin_kernel(win, numtaps, bands, bands.shape[0], scale)
    if scale:
        s = cupy.sum(hc)
        h /= s

        # Build up the coefficients.
        alpha = 0.5 * (numtaps - 1)
        m = cupy.arange(0, numtaps) - alpha
        h = 0
        for left, right in bands:
            h += right * cupy.sinc(right * m)
            h -= left * cupy.sinc(left * m)

        h *= win

        # Now handle scaling if desired.
        if scale:
            # Get the first passband.
            left, right = bands[0]
            if left == 0:
                scale_frequency = 0.0
            elif right == 1:
                scale_frequency = 1.0
            else:
                scale_frequency = 0.5 * (left + right)
            c = cupy.cos(cupy.pi * m * scale_frequency)
            s = cupy.sum(h * c)
            h /= s

    return h



def kaiser_atten(numtaps, width):
    """Compute the attenuation of a Kaiser FIR filter.

    Given the number of taps `N` and the transition width `width`, compute the
    attenuation `a` in dB, given by Kaiser's formula:

        a = 2.285 * (N - 1) * pi * width + 7.95

    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter.
    width : float
        The desired width of the transition region between passband and
        stopband (or, in general, at any discontinuity) for the filter,
        expressed as a fraction of the Nyquist frequency.

    Returns
    -------
    a : float
        The attenuation of the ripple, in dB.

    See Also
    --------
    scipy.signal.kaiser_atten
    """
    a = 2.285 * (numtaps - 1) * cupy.pi * width + 7.95
    return a



def kaiser_beta(a):
    """Compute the Kaiser parameter `beta`, given the attenuation `a`.

    Parameters
    ----------
    a : float
        The desired attenuation in the stopband and maximum ripple in
        the passband, in dB.  This should be a *positive* number.

    Returns
    -------
    beta : float
        The `beta` parameter to be used in the formula for a Kaiser window.

    References
    ----------
    Oppenheim, Schafer, "Discrete-Time Signal Processing", p.475-476.

    See Also
    --------
    scipy.signal.kaiser_beta

    """
    if a > 50:
        beta = 0.1102 * (a - 8.7)
    elif a > 21:
        beta = 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
    else:
        beta = 0.0
    return beta
