from __future__ import annotations
import functools
import typing

from os import cpu_count
from array_api_compat import is_cupy_array, is_torch_array
from math import ceil

from . import power_analysis
from .power_analysis import stat_ufunc_from_shorthand
from .util import (
    array_namespace,
    sliding_window_view,
    get_input_domain,
    Domain,
    _whichfloats,
    lazy_import,
    to_blocks,
    axis_index
)

from .type_stubs import ArrayType

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import scipy
    from scipy import special, signal
else:
    np = lazy_import('numpy')
    pd = lazy_import('pandas')
    scipy = lazy_import('scipy')
    special = lazy_import('scipy.special')
    signal = lazy_import('scipy.signal')

CPU_COUNT = cpu_count()
OLA_MAX_FFT_SIZE = 128 * 1024
INF = float('inf')


def _truncated_buffer(x: ArrayType, shape, dtype=None):
    if dtype is not None:
        x = x.view(dtype)
    out_size = np.prod(shape)
    assert x.size >= out_size
    return x.flatten()[: out_size].reshape(shape)


def fft(x, axis=-1, out=None, overwrite_x=False, plan=None, workers=None):
    if is_cupy_array(x):
        import cupy as cp

        # TODO: see about upstream question on this
        if out is None:
            pass
        else:
            out = out.reshape(x.shape)

        return cp.fft._fft._fftn(
            x,
            (None,),
            (axis,),
            None,
            cp.cuda.cufft.CUFFT_FORWARD,
            overwrite_x=overwrite_x,
            plan=plan,
            out=out,
            order='C',
        )
    else:
        if workers is None:
            workers = CPU_COUNT // 2
        return scipy.fft.fft(
            x, axis=axis, workers=workers, overwrite_x=overwrite_x, plan=plan
        )


def ifft(x, axis=-1, out=None, overwrite_x=False, plan=None, workers=None):
    if is_cupy_array(x):
        import cupy as cp

        # TODO: see about upstream question on this
        if out is None:
            pass
        else:
            out = out.reshape(x.shape)

        return cp.fft._fft._fftn(
            x,
            (None,),
            (axis,),
            None,
            cp.cuda.cufft.CUFFT_INVERSE,
            overwrite_x=overwrite_x,
            plan=plan,
            out=out,
            order='C',
        )
    else:
        if workers is None:
            workers = CPU_COUNT // 2
        return scipy.fft.ifft(
            x, axis=axis, workers=workers, overwrite_x=overwrite_x, plan=plan
        )


def fftfreq(n, d, *, xp=np, dtype='float64') -> ArrayType:
    """A replacement for `scipy.fft.fftfreq` that mitigates
    some rounding errors underlying `np.fft.fftfreq`.

    Further, no `fftshift` is needed for complex-valued data;
    the return result is monotonic beginning in the negative
    frequency half-space.

    Args:
        n: fft size
        d: sample spacing (inverse of sample rate)
        xp: the array module to use, dictating the return type

    Returns:
        an array of type `xp.ndarray`
    """
    dtype = np.dtype(dtype)
    fnyq = 1/(2*dtype.type(d))
    if n % 2 == 0:
        return xp.linspace(-fnyq, fnyq-2*fnyq/n, n, dtype=dtype)
    else:
        return xp.linspace(-fnyq+fnyq/n, fnyq-fnyq/n, n, dtype=dtype)


def zero_pad(x: ArrayType, pad_amt: int) -> ArrayType:
    """shortcut for e.g. np.pad(x, pad_amt, mode="constant", constant_values=0)"""
    xp = array_namespace(x)

    return xp.pad(x, pad_amt, mode='constant', constant_values=0)


def tile_axis0(x: ArrayType, N) -> ArrayType:
    """returns N copies of x along axis 0"""
    xp = array_namespace(x)
    return xp.tile(x.T, N).T


@functools.lru_cache(64)
def _get_window(name_or_tuple, N, fftbins=True, norm=True, *, dtype=None, xp=None):
    if xp is None:
        w = signal.windows.get_window(name_or_tuple, N, fftbins=fftbins)

        if norm:
            w /= np.sqrt(np.mean(np.abs(w) ** 2))
        return w
    else:
        w = _get_window(name_or_tuple, N)
        if hasattr(xp, 'asarray'):
            w = xp.asarray(w, dtype=dtype)
        else:
            w = xp.array(w).astype(dtype)
        return w


@functools.lru_cache
def equivalent_noise_bandwidth(window: str | tuple[str, float], N, fftbins=True):
    """return the equivalent noise bandwidth (ENBW) of a window, in bins"""
    w = _get_window(window, N, fftbins=fftbins)
    return len(w) * np.sum(w**2) / np.sum(w) ** 2


def broadcast_onto(a: ArrayType, other: ArrayType, axis: int) -> ArrayType:
    """broadcast a 1-D array onto a specified axis of `other`"""
    xp = array_namespace(a)

    slices = [xp.newaxis] * len(other.shape)
    slices[axis] = slice(None, None)
    return a.__getitem__(tuple(slices))


@functools.lru_cache(16)
def _get_stft_axes(
    fs: float, nfft: int, time_size: int, overlap_frac: float = 0, *, xp=np
) -> tuple[ArrayType, ArrayType]:
    """returns stft (freqs, times) array tuple appropriate to the array module xp"""

    freqs = fftfreq(nfft, 1/fs, xp=xp)
    times = xp.arange(time_size) * ((1 - overlap_frac) * nfft / fs)

    return freqs, times


@functools.lru_cache
def _prime_fft_sizes(min=2, max=OLA_MAX_FFT_SIZE):
    s = np.arange(3, max, 2)

    for m in range(3, int(np.sqrt(max) + 1), 2):
        if s[(m - 3) // 2]:
            s[(m * m - 3) // 2 :: m] = 0

    return s[(s > min)]


@functools.lru_cache
def design_cola_resampler(
    fs_base: float,
    fs_target: float,
    bw: float = INF,
    bw_lo: float = 0,
    min_oversampling: float = 1.1,
    min_fft_size=2 * 4096 - 1,
    shift=False,
    avoid_primes=True,
) -> tuple[float, float, dict]:
    """designs sampling and RF center frequency parameters that shift LO leakage outside of the specified bandwidth.

    The result includes the integer-divided SDR sample rate to request from the SDR, the LO frequency offset,
    and the keyword arguments needed to realize resampling with `ola_filter`.

    Args:
        fs_base: the base clock rate (sometimes known as master clock rate, MCR) of the receiver
        fs_target: the desired sample rate after resampling
        bw: the analysis bandwidth to protect from LO leakage
        bw_lo: the spectral leakage/phase noise bandwidth of the LO
        shift: the direction to shift the LO
        avoid_primes: whether to avoid large prime numbered FFTs for performance reasons

    Returns:
        (SDR sample rate, RF LO frequency offset in Hz, ola_filter_kws)
    """

    if bw == INF and shift:
        raise ValueError(
            'frequency shifting may only be applied when an analysis bandwidth is specified'
        )

    if shift:
        fs_sdr_min = fs_target + min_oversampling * bw / 2 + bw_lo / 2
    else:
        fs_sdr_min = fs_target

    if fs_base > fs_target:
        if shift and fs_sdr_min > fs_base:
            msg = f"""LO frequency shift with the requested parameters
            requires running the radio at a minimum {fs_sdr_min/1e6:0.2f} MS/s,
            but its maximum rate is {fs_base/1e6:0.2f} MS/s"""

            raise ValueError(msg)

        decimation = int(fs_base / fs_sdr_min)
        fs_sdr = fs_base / decimation
    else:
        fs_sdr = fs_base

    if bw != INF and bw > fs_base:
        raise ValueError(
            'passband bandwidth exceeds Nyquist bandwidth at maximum sample rate'
        )

    resample_ratio = fs_sdr / fs_target

    # the following returns the modulos closest to either 0 or 1, accommodating downward rounding errors (e.g., 0.999)
    trial_noverlap = resample_ratio * np.arange(1, OLA_MAX_FFT_SIZE + 1)
    check_mods = power_analysis.isroundmod(trial_noverlap, 1) & (
        trial_noverlap > min_fft_size * resample_ratio
    )

    # all valid noverlap size candidates
    valid_noverlap_out = 1 + np.where(check_mods)[0]
    if avoid_primes:
        reject = _prime_fft_sizes(100)
        valid_noverlap_out = np.setdiff1d(valid_noverlap_out, reject, True)
    if len(valid_noverlap_out) == 0:
        raise ValueError('no rational FFT sizes satisfied design constraints')

    nfft_out = valid_noverlap_out[0]
    nfft_in = int(np.rint(resample_ratio * nfft_out))

    if nfft_out % 2 == 1 or nfft_in % 2 == 1:
        nfft_out *= 2
        nfft_in *= 2

    # the following LO shift arguments assume that a hamming COLA window is used
    if shift == 'left':
        sign = -1
    elif shift == 'right':
        sign = +1
    elif shift in ('none', False, None):
        sign = 0
    else:
        raise ValueError(f'shift argument must be "left" or "right", not {repr(shift)}')

    if sign != 0 and bw == INF:
        raise ValueError('a passband bandwidth must be set to design a LO shift')

    if bw == INF:
        lo_offset = 0
        passband = (None, None)
    else:
        lo_offset = sign * (
            bw / 2 + bw_lo / 2
        )  # fs_sdr / nfft_in * (nfft_in - nfft_out)
        passband = (lo_offset - bw / 2, lo_offset + bw / 2)

    window = 'hamming'

    ola_resample_kws = {
        'window': window,
        'nfft': nfft_in,
        'nfft_out': nfft_out,
        'frequency_shift': shift,
        'passband': passband,
        'fs': fs_sdr,
    }

    return fs_sdr, lo_offset, ola_resample_kws


def _cola_scale(window, hop_size):
    if (window.size - hop_size) % 2 == 0:
        # scaling correction based on the shape of the window where it intersects with its neighbor
        cola_scale = 2 * window[(window.size - hop_size) // 2]
    else:
        cola_scale = (
            window[(window.size - hop_size) // 2]
            + window[(window.size - hop_size) // 2 + 1]
        )
    return cola_scale.real


def _stack_stft_windows(
    x: ArrayType,
    window: ArrayType,
    nperseg: int,
    noverlap: int,
    pad_mode='constant',
    norm=None,
    axis=0,
    out=None,
) -> ArrayType:
    """add overlapping windows at appropriate offset _to_overlapping_windows, returning a waveform.

    Compared to the underlying stft implementations in scipy and cupyx.scipy, this has been simplified
    to a reduced set of parameters for speed.

    Args:
        x: the 1-D waveform (or N-D tensor of waveforms)
        axis: the waveform axis; stft will be evaluated across all other axes
    """
    axis_slice = signal._arraytools.axis_slice

    nfft = nperseg
    hop_size = nperseg - noverlap

    strided = sliding_window_view(x, nfft, axis=axis)

    stride_windows = axis_slice(strided, start=0, step=hop_size, axis=axis)

    if norm is None:
        scale = _cola_scale(window, hop_size)
    elif norm == 'power':
        scale = 1
    else:
        raise ValueError(
            f"invalid normalization argument '{norm}' (should be 'cola' or 'psd')"
        )

    if out is None:
        out = stride_windows.copy()
    else:
        out = _truncated_buffer(out, stride_windows.shape)
        out[:] = stride_windows

    out *= broadcast_onto(window / scale, stride_windows, axis=axis + 1)

    return out


def _unstack_stft_windows(
    y: ArrayType, noverlap: int, nperseg: int, axis=0, out=None, extra=0
) -> ArrayType:
    """reconstruct the time-domain waveform from its STFT representation.

    Compared to the underlying istft implementations in scipy and cupyx.scipy, this has been simplified
    for speed at the expense of memory consumption.

    Args:
        y: the stft output, containing at least 2 dimensions
        noverlap: the overlap size that was used to generate the STFT (see scipy.signal.stft)
        axis: the axis of the first dimension of the STFT (the second is at axis+1)
        out: if specified, the output array that will receive the result. it must have at least the same allocated size as y
        extra: total number of extra samples to include at the edges
    """

    xp = array_namespace(y)
    axis_slice = signal._arraytools.axis_slice

    nfft = nperseg
    hop_size = nperseg - noverlap

    waveform_size = y.shape[axis] * y.shape[axis + 1] * hop_size // nfft + noverlap
    target_shape = y.shape[:axis] + (waveform_size,) + y.shape[axis + 2 :]

    if out is None:
        xr = xp.empty(target_shape, dtype=y.dtype)
    else:
        xr = _truncated_buffer(out, target_shape)

    xr_slice = axis_slice(
        xr,
        start=0,
        stop=noverlap,
        axis=axis,
    )
    xr_slice[:] = 0

    xr_slice = axis_slice(
        xr,
        start=-noverlap,
        stop=None,
        axis=axis,
    )
    xr_slice[:] = 0

    # for speed, sum up in groups of non-overlapping windows
    for offs in range(nfft // hop_size):
        yslice = axis_slice(y, start=offs, step=nfft // hop_size, axis=axis)
        yshape = yslice.shape

        yslice = yslice.reshape(
            yshape[:axis] + (yshape[axis] * yshape[axis + 1],) + yshape[axis + 2 :]
        )
        xr_slice = axis_slice(
            xr,
            start=offs * hop_size,
            stop=offs * hop_size + yslice.shape[axis],
            axis=axis,
        )

        if offs == 0:
            xr_slice[:] = yslice[: xr_slice.size]
        else:
            xr_slice += yslice[: xr_slice.size]

    return xr  # axis_slice(xr, start=noverlap-extra//2, stop=(-noverlap+extra//2) or None, axis=axis)


@functools.lru_cache
def _ola_filter_parameters(
    array_size: int, *, window, nfft_out: int, nfft: int, extend: bool
) -> tuple:
    if nfft_out is None:
        nfft_out = nfft

    if window == 'hamming':
        if nfft_out % 2 != 0:
            raise ValueError('blackman window COLA requires output nfft_out % 2 == 0')
        overlap_scale = 1 / 2
    elif window == 'blackman':
        if nfft_out % 3 != 0:
            raise ValueError('blackman window COLA requires output nfft_out % 3 == 0')
        overlap_scale = 2 / 3
    elif window == 'blackmanharris':
        if nfft_out % 5 != 0:
            raise ValueError('blackmanharris window requires output nfft_out % 5 == 0')
        overlap_scale = 4 / 5
    else:
        raise TypeError(
            'ola_filter argument "window" must be one of ("hamming", "blackman", or "blackmanharris")'
        )

    noverlap = round(nfft_out * overlap_scale)

    if array_size % noverlap != 0:
        if extend:
            pad_out = array_size % noverlap
        else:
            raise ValueError(
                f'x.size ({array_size}) is not an integer multiple of noverlap ({noverlap})'
            )
    else:
        pad_out = 0

    return nfft_out, noverlap, overlap_scale, pad_out


def _istft_buffer_size(
    array_size: int, *, window, nfft_out: int, nfft: int, extend: bool
):
    nfft_out, _, overlap_scale, pad_out = _ola_filter_parameters(**locals())
    nfft_max = max(nfft_out, nfft)
    fft_count = 2 + ((array_size + pad_out) / nfft_max) / overlap_scale
    size = ceil(fft_count * nfft_max)
    return size


def zero_stft_by_freq(
    freqs: ArrayType, xstft: ArrayType, *, passband: tuple[float, float], axis=0
) -> ArrayType:
    """apply a bandpass filter in the STFT domain by zeroing frequency indices"""
    xp = array_namespace(xstft)
    
    axis_slice = signal._arraytools.axis_slice

    freq_step = freqs[1] - freqs[0]
    fs = xstft.shape[axis] * freq_step
    ilo, ihi = _freq_band_edges(freqs.size, fs, *passband, xp=xp)
    axis_slice(xstft, start=0, stop=ilo, axis=axis + 1)[:] = 0
    axis_slice(xstft, start=ihi, stop=None, axis=axis + 1)[:] = 0
    return xstft


@functools.lru_cache(100)
def _find_downsample_copy_range(nfft_in: int, nfft_out: int, passband_start: int, passband_end: int):
    if passband_start is None:
        passband_start = 0
    if passband_end is None:
        passband_end = nfft_in
    passband_size = passband_end - passband_start
    passband_center = (passband_end + passband_start) // 2
    # passband_center_error = (passband_end - passband_start) % 2

    # copy input indexes, taken from the passband
    max_copy_size = min(passband_size, nfft_out)
    copy_in_start = max(passband_center - max_copy_size // 2, 0)
    copy_in_end = min(passband_center - max_copy_size // 2 + max_copy_size, nfft_in)
    copy_size = copy_in_end - copy_in_start

    assert copy_size <= nfft_out, (copy_size, nfft_out)
    assert copy_size >= 0, copy_size
    assert copy_in_end - copy_in_start == copy_size

    # copy output indexes
    output_zeros_size = max(nfft_out - copy_size, 0)
    copy_out_start = output_zeros_size // 2
    copy_out_end = copy_out_start + copy_size

    assert copy_out_end - copy_out_start == copy_size
    assert copy_out_start >= 0
    assert copy_out_end <= nfft_out

    return (copy_out_start, copy_out_end), (copy_in_start, copy_in_end), passband_center


@functools.lru_cache(16)
def _find_downsampled_freqs(nfft_out, freq_step, xp=np):
    return fftfreq(nfft_out, 1./(freq_step * nfft_out), xp=xp)


def downsample_stft(
    freqs: ArrayType,
    xstft: ArrayType,
    nfft_out: int,
    *,
    passband: tuple[float, float],
    axis=0,
    out=None,
) -> tuple[ArrayType, ArrayType]:
    """downsample and filter an STFT representation of a filter in the frequency domain.

    * This is rational downsampling by a factor of `nout/xstft.shape[axis+1]`,
      shifted if necessary to center the passband.
    * One approach to selecting `nfft_out` for this purpose is the use
      of `design_ola_filter`.

    Returns:
        A tuple containing the new `freqs` range and trimmed `xstft`
    """
    xp = array_namespace(xstft)
    axis_slice = signal._arraytools.axis_slice
    ax = axis + 1

    shape_out = list(xstft.shape)
    shape_out[ax] = nfft_out

    if out is None:
        xout = xp.empty(shape_out, dtype=xstft.dtype)
    else:
        xout = _truncated_buffer(out, shape_out, xstft.dtype)

    # passband indexes in the input
    freq_step = freqs[1] - freqs[0]
    fs = xstft.shape[ax] * freq_step
    passband_start, passband_end = _freq_band_edges(xstft.shape[ax], 1/fs, *passband)
    bounds_out, bounds_in, _ = _find_downsample_copy_range(xstft.shape[ax], nfft_out, passband_start, passband_end)
    freqs_out = _find_downsampled_freqs(nfft_out, freq_step, xp=xp)

    # copy first before zeroing, in case of input-output buffer reuse
    axis_slice(xout, *bounds_out, axis=ax)[:] = axis_slice(xstft, *bounds_in, axis=ax)
    axis_slice(xout, 0, bounds_out[0], axis=ax)[:] = 0
    axis_slice(xout, bounds_out[1], None, axis=ax)[:] = 0

    return freqs_out, xout


def stft(
    x: ArrayType,
    *,
    fs: float,
    window: ArrayType | str | tuple[str, float],
    nperseg: int = 256,
    noverlap: int = 0,
    axis: int = 0,
    truncate: bool = True,
    norm: str | None = None,
    out=None,
) -> tuple[ArrayType, ArrayType, ArrayType]:
    """Implements a stripped-down subset of scipy.fft.stft in order to avoid
    some overhead that comes with its generality and allow use of the generic
    python array-api for interchangable numpy/cupy support.

    For additional information, see help for scipy.fft.

    Args:
        x: input array

        fs: sampling rate

        window: a window array, or a name or (name, parameter) pair as in `scipy.signal.get_window`

        nperseg: the size of the FFT (= segment size used if overlapping)

        noverlap: if nonzero, compute windowed FFTs that overlap by this many bins (only 0 and nperseg//2 supported)

        axis: the axis on which to compute the STFT

        truncate: whether to allow truncation of `x` to enforce full fft block sizes

    Raises:
        NotImplementedError: if axis != 0

        ValueError: if truncate == False and x.shape[axis] % nperseg != 0

    Returns:
        stft (see scipy.fft.stft)

    """

    xp = array_namespace(x)

    # # For reference: this is probably the same
    # freqs, times, X = signal.spectral._spectral_helper(
    #     x,
    #     x,
    #     fs,
    #     window,
    #     nperseg,
    #     noverlap,
    #     nperseg,
    #     scaling="spectrum",
    #     axis=axis,
    #     mode="stft",
    #     padded=True,
    # )

    nfft = nperseg

    if norm not in ('power', None):
        raise TypeError('norm must be "power" or None')

    if isinstance(window, str) or (isinstance(window, tuple) and len(window) == 2):
        should_norm = norm == 'power'
        w = _get_window(window, nfft, xp=xp, dtype=x.dtype, norm=should_norm)
        if is_torch_array(w):
            import torch

            w = torch.asarray(w, dtype=x.dtype, device=x.device)
    elif window is None:
        w = xp.ones(nfft)
    else:
        w = xp.array(window)

    if noverlap == 0:
        x = to_blocks(x, nfft, axis=axis, truncate=truncate)

        x = x * broadcast_onto(w / nfft, x, axis=axis + 1)
        X = fft(x, axis=axis + 1, overwrite_x=True, out=out)

    else:
        x_ol = _stack_stft_windows(
            x,
            window=w / nfft,
            nperseg=nperseg,
            noverlap=noverlap,
            axis=axis,
            out=out,
            norm=norm,
        )

        X = fft(
            x_ol,
            axis=axis + 1,
            overwrite_x=True,
            out=None if out is None else _truncated_buffer(out, x_ol.shape),
        )

    # interleave the overlaps in time
    X = xp.fft.fftshift(X, axes=axis + 1)

    freqs, times = _get_stft_axes(
        fs,
        nfft=nfft,
        time_size=X.shape[axis],
        overlap_frac=noverlap / nfft,
        xp=np,
    )

    return freqs, times, X


def istft(
    xstft: ArrayType, size=None, *, nfft: int, noverlap: int, out=None, axis=0
) -> ArrayType:
    """reconstruct and return a waveform given its STFT and associated parameters"""

    xp = array_namespace(xstft)
    axis_slice = signal._arraytools.axis_slice

    x_windows = ifft(
        xp.fft.fftshift(xstft, axes=axis + 1),
        axis=axis + 1,
        overwrite_x=True,
        out=None if out is None else _truncated_buffer(out, xstft.shape),
    )

    y = _unstack_stft_windows(
        x_windows, noverlap=noverlap, nperseg=nfft, axis=axis, out=out
    )

    if size is not None:
        trim = y.shape[axis] - size
        if trim > 0:
            y = axis_slice(y, start=trim // 2, stop=-trim // 2, axis=axis)

    return y


def ola_filter(
    x: ArrayType,
    *,
    fs: float,
    nfft: int,
    window: str | tuple = 'hamming',
    passband: tuple[float, float],
    nfft_out: int = None,
    frequency_shift=False,
    axis=0,
    extend=False,
    out=None,
):
    """apply a bandpass filter implemented through STFT overlap-and-add.

    Args:
        x: the input waveform
        fs: the sample rate of the input waveform, in Hz
        noverlap: the size of overlap between adjacent FFT windows, in samples
        window: the type of COLA window to apply, 'hamming', 'blackman', or 'blackmanharris'
        passband: a tuple of low-pass cutoff and high-pass cutoff frequency (or None to skip either)
        nfft_out: implement downsampling by adjusting the size of overlap between adjacent FFT windows
        frequency_shift: the direction to shift the downsampled frequencies ('left' or 'right', or False to center)
        axis: the axis of `x` along which to compute the filter
        extend: if True, allow use of zero-padded samples at the edges to accommodate a non-integer number of overlapping windows in x
        out: None, 'shared', or an array object to receive the output data

    Returns:
        an Array of the same shape as X
    """
    xp = array_namespace(x)

    nfft_out, noverlap, overlap_scale, _ = _ola_filter_parameters(
        x.size,
        window=window,
        nfft_out=nfft_out,
        nfft=nfft,
        extend=extend,
    )

    enbw = equivalent_noise_bandwidth(window, nfft_out, fftbins=False)

    w = _get_window(window, nfft, fftbins=False, xp=xp)

    freqs, _, xstft = stft(
        x,
        fs=fs,
        window=w,
        nperseg=nfft,
        noverlap=round(nfft * overlap_scale),
        axis=axis,
        truncate=False,
        out=out,
    )

    zero_stft_by_freq(
        freqs, xstft, passband=(passband[0] + enbw, passband[1] - enbw), axis=axis
    )

    if nfft_out != nfft or frequency_shift:
        freqs, xstft = downsample_stft(
            freqs,
            xstft,
            nfft_out=nfft_out,
            passband=passband,
            axis=axis,
            out=xstft,
        )

    return istft(
        xstft,
        round(x.shape[axis] * nfft_out / nfft),
        nfft=nfft_out,
        noverlap=noverlap,
        out=out,
        axis=axis,
    )


@functools.lru_cache
def _freq_band_edges(n, d, cutoff_low, cutoff_hi, *, xp=np):
    freqs = fftfreq(n, d, xp=xp)

    if cutoff_low is None:
        ilo = None
    else:
        ilo = xp.where(freqs >= cutoff_low)[0][0]

    if cutoff_hi is None:
        ihi = None
    elif cutoff_hi >= freqs[-1]:
        ihi = freqs.size
    else:
        ihi = xp.where(freqs <= cutoff_hi)[0][-1]

    return ilo, ihi


def spectrogram(
    x: ArrayType,
    *,
    fs: float,
    window: ArrayType | str | tuple[str, float],
    nperseg: int = 256,
    noverlap: int = 0,
    axis: int = 0,
    truncate: bool = True,
    out=None,
):
    kws = dict(locals())

    freqs, times, X = stft(norm='power', **kws)
    spg = power_analysis.envtopow(X, out=X).real

    return freqs, times, spg


def persistence_spectrum(
    x: ArrayType,
    *,
    fs: float,
    bandwidth=INF,
    window,
    resolution: float,
    fractional_overlap=0,
    statistics: list[float],
    truncate=True,
    dB=True,
    axis=0,
) -> ArrayType:
    # TODO: support other persistence statistics, such as mean

    if power_analysis.isroundmod(fs, resolution):
        nfft = round(fs / resolution)
        noverlap = round(fractional_overlap * nfft)
    else:
        # need sample_rate_Hz/resolution to give us a counting number
        raise ValueError('sample_rate_Hz/resolution must be a counting number')

    xp = array_namespace(x)
    axis_slice = signal._arraytools.axis_slice
    domain = get_input_domain()

    if domain == Domain.TIME:
        freqs, _, X = spectrogram(
            x, window=window, fs=fs, nperseg=nfft, noverlap=noverlap, axis=axis
        )
    elif domain == Domain.FREQUENCY:
        X = x
        freqs, _ = _get_stft_axes(
            fs=fs,
            nfft=nfft,
            time_size=X.shape[axis],
            overlap_frac=noverlap / nfft,
            xp=np,
        )
    else:
        raise ValueError('unsupported persistence spectrum domain "{domain}')

    if truncate:
        if bandwidth == INF:
            bw_args = (None, None)
        else:
            bw_args = (-bandwidth / 2, +bandwidth / 2)
        ilo, ihi = _freq_band_edges(freqs.size, 1.0/fs, *bw_args)
        X = X[:, ilo:ihi]

    if domain == Domain.TIME:
        if dB:
            spg = power_analysis.powtodB(X, eps=1e-25, out=X.real)
        else:
            spg = X.astype('float32')
    elif domain == Domain.FREQUENCY:
        if dB:
            # here X is complex-valued; use the first-half of its buffer
            spg = power_analysis.envtodB(X, eps=1e-25, out=X.real)
        else:
            spg = power_analysis.envtopow(X, out=X.real)
    else:
        raise ValueError(f'unhandled dB and domain: {dB}, {domain}')

    isquantile = _whichfloats(tuple(statistics))

    shape = list(spg.shape)
    shape[axis] = len(statistics)
    out = xp.empty(tuple(shape), dtype='float32')

    quantiles = list(np.asarray(statistics)[isquantile].astype('float32'))

    out_quantiles = axis_index(out, isquantile, axis=axis).swapaxes(0,1)
    out_quantiles[:] = xp.quantile(
        spg, xp.array(quantiles), axis=axis
    )

    for i, isquantile in enumerate(isquantile):
        if not isquantile:
            ufunc = stat_ufunc_from_shorthand(statistics[i], xp=xp)
            axis_index(out, i, axis=axis)[...] = ufunc(spg, axis=axis)

    return out


def channelize_power(
    iq: ArrayType,
    Ts: float,
    fft_size_per_channel: int,
    *,
    analysis_bins_per_channel: int,
    window: ArrayType,
    fft_overlap_per_channel=0,
    channel_count: int = 1,
    axis=0,
):
    """Channelizes the input waveform and returns a time series of power in each channel.

    The calculation is performed by transformation into the frequency domain. Power is
    summed across the bins in the analysis bandwidth, ignoring those in bins outside
    of the analysis bandwidth.

    The total analysis bandwidth (i.e., covering all channels) is equal to
    `(analysis_bins_per_channel/fft_size_per_channel)/Ts`,
    occupying the center of the total sampling bandwidth. The bandwidth in each power bin is equal to
    `(analysis_bins_per_channel/fft_size_per_channel)/Ts/channel_count`.

    The time spacing of the power samples is equal to `Ts * fft_size_per_channel * channel_count`
    if `fft_overlap_per_channel` is 0, otherwise, `Ts * fft_size_per_channel * channel_count / 2`.

    Args:
        iq: an input waveform or set of input waveforms, with time along axis 0

        Ts: the sampling period (1/sampling_rate)

        fft_size_per_channel: the size of the fft to use in each channel; total fft size is (channel_count * fft_size_per_channel)

        channel_count: the number of channels to analyze

        fft_overlap_per_channel: equal to 0 to disable overlapping windows, or to disable overlap, or fft_size_per_channel // 2)

        analysis_bins_per_channel: the number of bins to keep in each channel

        window: callable window function to use in the analysis

        axis: the axis along which to perform the FFT (for now, require axis=0)

    Raises:
        NotImplementedError: if axis != 0

        NotImplementedError: if fft_overlap_per_channel is not one of (0, fft_size_per_channel//2)

        ValueError: if analysis_bins_per_channel > fft_size_per_channel

        ValueError: if channel_count * (fft_size_per_channel - analysis_bins_per_channel) is not even
    """
    if axis != 0:
        raise NotImplementedError('sorry, only axis=0 implemented for now')

    if analysis_bins_per_channel > fft_size_per_channel:
        raise ValueError('the number of analysis bins cannot be greater than FFT size')

    freqs, times, X = stft(
        iq,
        fs=1.0 / Ts,
        w=window,
        nperseg=fft_size_per_channel * channel_count,
        noverlap=fft_overlap_per_channel * channel_count,
        norm='power',
        axis=axis,
    )

    # extract only the bins inside the analysis bandwidth
    skip_bins = channel_count * (fft_size_per_channel - analysis_bins_per_channel)
    if skip_bins % 2 == 1:
        raise ValueError('must pass an even number of bins to skip')
    X = X[:, skip_bins // 2 : -skip_bins // 2]
    freqs = freqs[skip_bins // 2 : -skip_bins // 2]

    if channel_count == 1:
        channel_power = power_analysis.envtopow(X).sum(axis=axis + 1)

        return times, channel_power

    else:
        freqs = to_blocks(freqs, analysis_bins_per_channel)
        X = to_blocks(X, analysis_bins_per_channel, axis=axis + 1)

        channel_power = power_analysis.envtopow(X).sum(axis=axis + 2)

        return freqs[0], times, channel_power


def iq_to_stft_spectrogram(
    iq: ArrayType,
    window: ArrayType | str | tuple[str, float],
    nfft: int,
    Ts,
    overlap=True,
    analysis_bandwidth=None,
):
    xp = array_namespace(iq)

    freqs, times, X = stft(
        iq,
        fs=1.0 / Ts,
        window=window,
        nperseg=nfft,
        noverlap=nfft // 2 if overlap else 0,
        norm='power',
        axis=0,
    )

    # X = xp.fft.fftshift(X, axes=0)/xp.sqrt(nfft*Ts)
    X = power_analysis.envtopow(X)

    spg = pd.DataFrame(X, columns=freqs, index=times)

    if analysis_bandwidth is not None:
        throwaway = spg.shape[1] * (
            1 - analysis_bandwidth * Ts
        )  # (len(freqs)-int(xp.rint(nfft*analysis_bandwidth*Ts)))//2
        if len(times) > 1 and xp.abs(throwaway - xp.rint(throwaway)) > 1e-6:
            raise ValueError(
                f'analysis bandwidth yield integral number of samples, but got {throwaway}'
            )
        # throwaway = throwaway
        # if throwaway % 2 == 1:
        #     raise ValueError('should have been even')
        spg = spg.iloc[:, int(xp.floor(throwaway / 2)) : -int(xp.ceil(throwaway // 2))]

    return spg


def time_to_frequency(iq, Ts, window=None, axis=0):
    xp = array_namespace(iq)

    if window is None:
        window = signal.windows.blackmanharris(iq.shape[0], sym=False)

    window /= iq.shape[0] * xp.sqrt((window).mean())
    window = broadcast_onto(window, iq, axis=0)

    X = xp.fft.fftshift(
        fft(iq * window, axis=0),
        axes=0,
    )
    fftfreqs = fftfreq(X.shape[0], Ts, xp=xp)
    return fftfreqs, X
