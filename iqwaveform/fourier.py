from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import signal, special
from . import power_analysis
import scipy
from multiprocessing import cpu_count
from functools import partial, lru_cache
from .util import (
    Array,
    array_stream,
    array_namespace,
    pad_along_axis,
    sliding_window_view,
    get_input_domain,
    Domain,
    float_dtype_like,
    empty_shared,
    _whichfloats,
)
from array_api_compat import is_cupy_array, is_torch_array
from scipy.signal._arraytools import axis_slice
from .power_analysis import stat_ufunc_from_shorthand


CPU_COUNT = cpu_count()
OLA_MAX_FFT_SIZE = 64 * 1024
PAD_WINDOWS = 3


def _is_shared_arg(arg):
    if not isinstance(arg, str):
        return False

    if arg == 'shared':
        return True
    else:
        raise ValueError('"shared" is the only valid string argument for out')


def _truncated_buffer(x: Array, shape):
    return x.flatten()[: np.prod(shape)].reshape(shape)


def fft(x, axis=-1, out=None, overwrite_x=False, plan=None, workers=None):
    if is_cupy_array(x):
        import cupy as cp

        # TODO: see about upstream question on this
        if out is None:
            pass
        elif _is_shared_arg(out):
            out = empty_shared(target_shape, dtype, xp=cp)
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
        elif _is_shared_arg(out):
            out = empty_shared(target_shape, dtype, xp=cp)
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


def zero_pad(x: Array, pad_amt: int) -> Array:
    """shortcut for e.g. np.pad(x, pad_amt, mode="constant", constant_values=0)"""
    xp = array_namespace(x)

    return xp.pad(x, pad_amt, mode='constant', constant_values=0)


def tile_axis0(x: Array, N) -> Array:
    """returns N copies of x along axis 0"""
    xp = array_namespace(x)
    return xp.tile(x.T, N).T


def to_blocks(y: Array, size: int, truncate=False, axis=0) -> Array:
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


@lru_cache(8)
def _get_window(name_or_tuple, N, norm=True, dtype=None, xp=None):
    if xp is None:
        w = signal.windows.get_window(name_or_tuple, N)

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


@lru_cache
def equivalent_noise_bandwidth(window: str | tuple[str, float], N):
    """return the equivalent noise bandwidth (ENBW) of a window, in bins"""
    w = _get_window(window, N)
    return len(w) * np.sum(w**2) / np.sum(w) ** 2


def broadcast_onto(a: Array, other: Array, axis: int) -> Array:
    """broadcast a 1-D array onto a specified axis of `other`"""
    xp = array_namespace(a)

    slices = [xp.newaxis] * len(other.shape)
    slices[axis] = slice(None, None)
    return a.__getitem__(tuple(slices))


@lru_cache(16)
def _get_stft_axes(
    fs: float, fft_size: int, time_size: int, overlap_frac: float = 0, xp=np
) -> tuple[Array, Array]:
    """returns stft (freqs, times) array tuple appropriate to the array module xp"""

    freqs = xp.fft.fftshift(xp.fft.fftfreq(fft_size, d=1 / fs))
    times = xp.arange(time_size) * ((1 - overlap_frac) * fft_size / fs)

    return freqs, times


@lru_cache
def _prime_fft_sizes(min=2, max=OLA_MAX_FFT_SIZE):
    s = np.arange(3, max, 2)

    for m in range(3, int(np.sqrt(max) + 1), 2):
        if s[(m - 3) // 2]:
            s[(m * m - 3) // 2 :: m] = 0

    return s[(s > min)]


@lru_cache
def design_cola_resampler(
    fs_base: float,
    fs_target: float,
    bw: float,
    bw_lo: float = 250e3,
    min_oversampling: float = 1.1,
    min_fft_size=2 * 4096,
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

    if shift:
        fs_sdr_min = fs_target + min_oversampling * bw / 2 + bw_lo / 2
    else:
        fs_sdr_min = fs_target

    decimation = int(fs_base / fs_sdr_min)

    fs_sdr = fs_base / decimation

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

    # the following LO shift arguments assume that a hamming COLA window is used
    if shift == 'left':
        sign = -1
    elif shift == 'right':
        sign = +1
    elif shift in ('none', False, None):
        sign = 0
    else:
        raise ValueError(f'shift argument must be "left" or "right", not {repr(shift)}')

    lo_offset = sign * (bw / 2 + bw_lo / 2)  # fs_sdr / nfft_in * (nfft_in - nfft_out)

    window = 'hamming'
    # enbw = (fs_target / nfft_out) * equivalent_noise_bandwidth(window, nfft_out)

    # if bw is None:
    #     passband = (None, None)
    # else:
    passband = (lo_offset - (bw-enbw) / 2, lo_offset + (bw-enbw) / 2)

    ola_resample_kws = {
        'window': window,
        'fft_size': nfft_in,
        'fft_size_out': nfft_out,
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
    x: Array,
    window: Array,
    nperseg: int,
    noverlap: int,
    pad_mode='constant',
    axis=0,
    out=None,
) -> Array:
    """add overlapping windows at appropriate offset _to_overlapping_windows, returning a waveform.

    Compared to the underlying stft implementations in scipy and cupyx.scipy, this has been simplified
    to a reduced set of parameters for speed.

    Args:
        x: the 1-D waveform (or N-D tensor of waveforms)
        axis: the waveform axis; stft will be evaluated across all other axes
    """
    xp = array_namespace(x)

    fft_size = nperseg
    hop_size = nperseg - noverlap

    x = pad_along_axis(
        x, [PAD_WINDOWS * noverlap, PAD_WINDOWS * noverlap], mode=pad_mode, axis=axis
    )

    strided = sliding_window_view(x, fft_size, axis=axis)

    stride_windows = axis_slice(strided, start=0, step=hop_size, axis=axis)

    cola_scale = _cola_scale(window, hop_size)

    if out is None:
        out = stride_windows.copy()
    elif _is_shared_arg(out):
        out = empty_shared(stride_windows.shape, stride_windows.dtype, xp=xp)
        out[:] = stride_windows
    else:
        out = _truncated_buffer(out, stride_windows.shape)
        out[:] = stride_windows

    out *= broadcast_onto(window / cola_scale, stride_windows, axis=axis + 1)

    return out


def _unstack_stft_windows(
    y: Array, noverlap: int, nperseg: int, axis=0, out=None, extra=0
) -> Array:
    """reconstruct the time-domain waveform from the stft in y.

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

    fft_size = nperseg
    hop_size = nperseg - noverlap

    waveform_size = y.shape[axis] * y.shape[axis + 1] * hop_size // fft_size + noverlap
    target_shape = y.shape[:axis] + (waveform_size,) + y.shape[axis + 2 :]

    if out is None:
        xr = xp.empty(target_shape, dtype=y.dtype)
    elif _is_shared_arg(out):
        xr = empty_shared(target_shape, dtype, xp)
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
    for offs in range(fft_size // hop_size):
        yslice = axis_slice(y, start=offs, step=fft_size // hop_size, axis=axis)
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


@lru_cache
def _ola_filter_parameters(
    array_size: int, *, window, fft_size_out: int, fft_size: int, extend: bool
) -> tuple:
    if fft_size_out is None:
        fft_size_out = fft_size

    if fft_size < fft_size_out:
        raise ValueError('only downsampling is supported')

    if window == 'hamming':
        if fft_size_out % 2 != 0:
            raise ValueError(
                'blackman window COLA requires output fft_size_out % 2 == 0'
            )
        overlap_scale = 1 / 2
    elif window == 'blackman':
        if fft_size_out % 3 != 0:
            raise ValueError(
                'blackman window COLA requires output fft_size_out % 3 == 0'
            )
        overlap_scale = 2 / 3
    elif window == 'blackmanharris':
        if fft_size_out % 5 != 0:
            raise ValueError(
                'blackmanharris window requires output fft_size_out % 5 == 0'
            )
        overlap_scale = 4 / 5
    else:
        raise TypeError(
            'ola_filter argument "window" must be one of ("hamming", "blackman", or "blackmanharris")'
        )

    noverlap = round(fft_size_out * overlap_scale)

    if array_size % noverlap != 0:
        if extend:
            pad_out = array_size % noverlap
        else:
            raise ValueError(
                f'x.size ({array_size}) is not an integer multiple of noverlap ({noverlap})'
            )
    else:
        pad_out = 0

    return fft_size_out, noverlap, overlap_scale, pad_out


def _istft_buffer_size(
    array_size: int, *, window, fft_size_out: int, fft_size: int, extend: bool
):
    fft_size_out, noverlap, overlap_scale, pad_out = _ola_filter_parameters(**locals())
    N = round(
        np.ceil(((array_size + pad_out) / fft_size + 2 * PAD_WINDOWS) / overlap_scale)
        * fft_size
    )
    return N


def zero_stft_by_freq(
    freqs: Array, xstft: Array, *, passband: tuple[float, float], axis=0
) -> Array:
    """apply a bandpass filter in the STFT domain by zeroing frequency indices"""

    ilo, ihi = _freq_band_edges(freqs[0], freqs[-1], freqs.size, *passband)
    axis_slice(xstft, start=0, stop=ilo, axis=axis + 1)[:] = 0
    axis_slice(xstft, start=ihi, stop=None, axis=axis + 1)[:] = 0
    return xstft


def downsample_stft(
    freqs: Array,
    xstft: Array,
    fft_size_out: int,
    *,
    passband: tuple[float, float],
    axis=0,
    out=None,
) -> tuple[Array, Array]:
    """downsample and filter an STFT representation of a filter in the frequency domain.

    * This is rational downsampling by a factor of `nout/xstft.shape[axis+1]`,
      shifted if necessary to center the passband.
    * One approach to selecting `fft_size_out` for this purpose is the use
      of `design_ola_filter`.

    Returns:
        A tuple containing the new `freqs` range and trimmed `xstft`
    """
    xp = array_namespace(xstft)

    ilo, ihi = _freq_band_edges(freqs[0], freqs[-1], freqs.size, *passband)
    pass_size = ihi - ilo
    pad_size = fft_size_out - pass_size

    ioutlo = pad_size // 2
    iouthi = fft_size_out - pad_size // 2 - pad_size % 2

    if out is None:
        shape = list(xstft.shape)
        shape[axis + 1] = fft_size_out
        xout = xp.empty(shape)
    else:
        xout = out

    # truncate to the specified range of frequency bins
    freqs_out = freqs[ioutlo:iouthi][:fft_size_out]

    axis_slice(xout, ioutlo, iouthi, axis=axis + 1)[:] = axis_slice(
        xout, ilo, ihi, axis=axis + 1
    )
    xout = axis_slice(xout, 0, fft_size_out, axis=axis + 1)

    return freqs_out, xout


def stft(
    x: Array,
    *,
    fs: float,
    window: Array | str | tuple[str, float],
    nperseg: int = 256,
    noverlap: int = 0,
    axis: int = 0,
    truncate: bool = True,
    norm: str | None = None,
    out=None,
) -> tuple[np.array, np.array, Array]:
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

    fft_size = nperseg

    if norm not in ('power', None):
        raise TypeError('norm must be "power" or None')

    if isinstance(window, str) or (isinstance(window, tuple) and len(window) == 2):
        should_norm = norm == 'power'
        w = _get_window(window, fft_size, xp=xp, dtype=x.dtype, norm=should_norm)
        if is_torch_array(w):
            import torch

            w = torch.asarray(w, dtype=x.dtype, device=x.device)
    else:
        w = xp.array(window)

    if noverlap == 0:
        x = to_blocks(x, fft_size, truncate=truncate)

        x = x * broadcast_onto(w / fft_size, x, axis=axis + 1)
        X = fft(x, axis=axis + 1, overwrite_x=True, out=out)
        X = xp.fft.fftshift(X, axes=axis + 1)

    else:
        # assert fft_size % (fft_size - noverlap) == 0

        x_ol = _stack_stft_windows(
            x, window=w, nperseg=nperseg, noverlap=noverlap, axis=axis, out=out
        )

        X = fft(x_ol, axis=axis + 1, overwrite_x=True, out=x_ol)

        # interleave the overlaps in time
        # X = X.reshape(X.shape[:-2] + (X.shape[-2]*X.shape[-1],))
        X = xp.fft.fftshift(X, axes=axis + 1)
        # X = X.swapaxes(-2, -1)

    freqs, times = _get_stft_axes(
        fs,
        fft_size=fft_size,
        time_size=X.shape[axis],
        overlap_frac=noverlap / fft_size,
        xp=np,
    )

    return freqs, times, X


def istft(
    xstft: Array, size, *, fft_size: int, noverlap: int, out=None, cache=None, axis=0
) -> Array:
    """reconstruct and return a waveform given its STFT and associated parameters"""

    xp = array_namespace(xstft)

    x_windows = ifft(
        xp.fft.fftshift(xstft, axes=axis + 1),
        axis=axis + 1,
        overwrite_x=True,
        # out=X if cache is None else None
    )

    if cache is not None:
        cache['stft'] = xstft
    elif out is None or isinstance(out, str) and out == 'shared':
        out = xstft

    y = _unstack_stft_windows(
        x_windows, noverlap=noverlap, nperseg=fft_size, axis=axis, out=out
    )

    trim = y.shape[axis] - round(size * fft_size / fft_size)
    if trim > 0:
        y = axis_slice(y, start=trim // 2, stop=(-trim // 2) or None, axis=axis)

    return y


def ola_filter(
    x: Array,
    *,
    fs: float,
    fft_size: int,
    window: str | tuple = 'hamming',
    passband: tuple[float, float],
    fft_size_out: int = None,
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
        fft_size_out: implement downsampling by adjusting the size of overlap between adjacent FFT windows
        frequency_shift: the direction to shift the downsampled frequencies ('left' or 'right', or False to center)
        axis: the axis of `x` along which to compute the filter
        extend: if True, allow use of zero-padded samples at the edges to accommodate a non-integer number of overlapping windows in x
        out: None, 'shared', or an array object to receive the output data

    Returns:
        an Array of the same shape as X
    """
    fft_size_out, noverlap, overlap_scale, _ = _ola_filter_parameters(
        x.size,
        window=window,
        fft_size_out=fft_size_out,
        fft_size=fft_size,
        extend=extend,
    )

    enbw = equivalent_noise_bandwidth(window, fft_size_out)
    passband = (passband[0] + enbw/2, passband[1] - enbw/2)

    freqs, _, xstft = stft(
        x,
        fs=fs,
        window=window,
        nperseg=fft_size,
        noverlap=round(fft_size * overlap_scale),
        axis=axis,
        truncate=False,
        # out=out
    )

    if fft_size_out != fft_size or frequency_shift:
        freqs, xstft = downsample_stft(
            freqs, xstft, fft_size_out=fft_size_out, passband=passband, axis=axis, out=xstft
        )

    zero_stft_by_freq(freqs, xstft, passband=passband, axis=axis)

    return istft(
        xstft,
        x.shape[axis],
        fft_size=fft_size_out,
        noverlap=noverlap,
        out=out,
        axis=axis,
    )


@lru_cache(8)
def _freq_band_edges(freq_min, freq_max, freq_count, cutoff_low, cutoff_hi):
    freq_inds = np.linspace(freq_min, freq_max, freq_count)

    if cutoff_low is None:
        ilo = None
    else:
        ilo = np.where(freq_inds >= cutoff_low)[0][0]

    if cutoff_hi is None:
        ihi = None
    else:
        ihi = np.where(freq_inds <= cutoff_hi)[0][-1] + 1

    return ilo, ihi


def spectrogram(
    x: Array,
    *,
    fs: float,
    window: Array | str | tuple[str, float],
    nperseg: int = 256,
    noverlap: int = 0,
    axis: int = 0,
    truncate: bool = True,
    norm: str | None = None,
    out=None,
):
    kws = dict(locals())

    if is_cupy_array(x):
        from . import cuda

        cuda.build()

        with cuda.apply_abs2_in_fft:
            freqs, times, spg = stft(**kws)
            spg = spg.real

    else:
        freqs, times, X = stft(**kws)
        spg = power_analysis.envtopow(X)

    return freqs, times, spg


def persistence_spectrum(
    x: Array,
    *,
    fs: float,
    bandwidth=None,
    window,
    resolution: float,
    fractional_overlap=0,
    statistics: list[float],
    truncate=True,
    dB=True,
    axis=0,
) -> Array:
    # TODO: support other persistence statistics, such as mean

    if power_analysis.isroundmod(fs, resolution):
        fft_size = round(fs / resolution)
        noverlap = round(fractional_overlap * fft_size)
    else:
        # need sample_rate_Hz/resolution to give us a counting number
        raise ValueError('sample_rate_Hz/resolution must be a counting number')

    xp = array_namespace(x)
    domain = get_input_domain()
    dtype = float_dtype_like(x)

    if domain == Domain.TIME:
        freqs, _, X = spectrogram(
            x, window=window, fs=fs, nperseg=fft_size, noverlap=noverlap, axis=axis
        )
    elif domain == Domain.FREQUENCY:
        X = x
        freqs, _ = _get_stft_axes(
            fs=fs,
            fft_size=fft_size,
            time_size=X.shape[axis],
            overlap_frac=noverlap / fft_size,
            xp=np,
        )
    else:
        raise ValueError('unsupported persistence spectrum domain "{domain}')

    if truncate:
        if bandwidth is None:
            bw_args = (None, None)
        else:
            bw_args = (-bandwidth / 2, +bandwidth / 2)
        ilo, ihi = _freq_band_edges(freqs[0], freqs[-1], freqs.size, *bw_args)
        X = X[:, ilo:ihi]

    if domain == Domain.TIME and dB:
        # already power
        spg = power_analysis.powtodB(X, eps=1e-25, out=X)
    elif domain == Domain.FREQUENCY and dB:
        # here X is complex-valued; use the first-half of its buffer
        spg = power_analysis.envtodB(X, eps=1e-25)
    elif domain == Domain.FREQUENCY and not dB:
        spg = power_analysis.envtopow(X, eps=1e-25)

    isquantile = _whichfloats(tuple(statistics))

    shape = list(spg.shape)
    shape[axis] = len(statistics)
    out = xp.empty(tuple(shape))

    quantiles = list(np.asarray(statistics)[isquantile].astype('float32'))

    # TODO: access the proper axis of spg in the output buffer
    out[isquantile] = xp.quantile(
        spg, xp.array(quantiles), axis=axis, out=out[isquantile]
    )

    for i, isquantile in enumerate(isquantile):
        if not isquantile:
            ufunc = stat_ufunc_from_shorthand(statistics[i], xp=xp)
            axis_slice(out, start=i, stop=i + 1, axis=axis)[...] = ufunc(spg, axis=axis)

    return out


def low_pass_filter(
    iq: Array,
    Ts: float,
    bandwidth: float,
    window: Array = None,
    fc_offset: float = 0,
    axis=0,
) -> Array:
    """Applies a low-pass filter to the input waveform by conversion into the frequency domain.

    Args:
        iq: the (optionally complex-valued) waveform to be filtered, with shape (N0, ..., N[K-1])

        Ts: sampling period (`1/sampling_rate`)

        bandwidth: the bandwidth of the signal. this represents the bins to be zeroed in the FFT domain.

        window: the windowing function to apply to IQ. ()

    Returns:
        np.ndarray with shape (N0, ..., N[K-1])
    """

    xp = array_namespace(iq, window)

    if window is None:
        if axis != 0 or len(iq.shape) != 2:
            raise NotImplementedError(
                f'in current implementation, `window` can only be used when axis=0 and iq has 2 axes'
            )
        window = xp.array([1])

    X = xp.fft.fftshift(
        fft(
            iq * window[:, xp.newaxis],
            axis=axis,
        ),
        axes=axis,
    )
    fftfreqs = xp.fft.fftshift(xp.fft.fftfreq(X.shape[0], Ts))

    freq_spacing = fftfreqs[1] - fftfreqs[0]
    inband_mask = xp.abs(fftfreqs + fc_offset + freq_spacing / 2) < bandwidth / 2
    inband = xp.where(inband_mask)[0]
    inband_offset = iq.shape[0] // 2 - int(xp.rint(inband.mean()))

    center = xp.where(inband_mask)[0] - inband_offset
    outside = xp.where(~inband_mask)[0] - inband_offset

    # ind_offset = inds[0]-X.shape[0]//2
    X[center], X[outside] = X[inband], 0

    x = (
        ifft(
            xp.fft.fftshift(X, axes=axis),
            axis=axis,
        )
        / window[:, xp.newaxis]
    )

    return x  # .astype('complex64')


def upsample(iq: Array, factor: int, Ts: float = None, shift_bins=0, axis=0):
    """Upsamples a signal by an integer factor, low-pass filtered so that the new higher frequencies are empty.

    Implementation is by zero-padding in the Fourier domain.

    Args:

        iq: input waveform, complex- or real-valued

        upsample_factor: defined such that the output sample period is `Ts/upsample_factor`

        Ts: sample period

        shift_bins: shift the zero-padded FFT by this many bins.

    Returns:

        (iq_upsampled) if Ts is None, otherwise (iq_upsampled, Ts/upsample_factor)

    """

    xp = array_namespace(iq)

    X = xp.fft.fftshift(
        fft(iq * factor, axis=axis),
        axes=axis,
    )

    if factor == 1:
        if Ts is None:
            return iq
        else:
            return iq, Ts

    count = iq.shape[0] * (factor - 1) / 2

    X = zero_pad(
        X,
        [[int(xp.floor(count)) - shift_bins, int(xp.ceil(count)) + shift_bins]]
        + [[0, 0]] * (len(iq.shape) - 1),
    )
    x = ifft(
        xp.fft.fftshift(X, axes=axis),
        axis=axis,
    )

    if Ts is None:
        return x  # [iq.shape[0] : 2 * iq.shape[0]]
    else:
        return x, Ts / factor  # [iq.shape[0] : 2 * iq.shape[0]], Ts / factor


def channelize_power(
    iq: Array,
    Ts: float,
    fft_size_per_channel: int,
    *,
    analysis_bins_per_channel: int,
    window: Array,
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
        raise ValueError(f'the number of analysis bins cannot be greater than FFT size')

    xp = array_namespace(iq)

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
    iq: Array,
    window: Array | str | tuple[str, float],
    fft_size: int,
    Ts,
    overlap=True,
    analysis_bandwidth=None,
):
    xp = array_namespace(iq)

    freqs, times, X = stft(
        iq,
        fs=1.0 / Ts,
        window=window,
        nperseg=fft_size,
        noverlap=fft_size // 2 if overlap else 0,
        norm='power',
        axis=0,
    )

    # X = xp.fft.fftshift(X, axes=0)/xp.sqrt(fft_size*Ts)
    X = power_analysis.envtopow(X)

    spg = pd.DataFrame(X, columns=freqs, index=times)

    if analysis_bandwidth is not None:
        throwaway = spg.shape[1] * (
            1 - analysis_bandwidth * Ts
        )  # (len(freqs)-int(xp.rint(FFT_SIZE*analysis_bandwidth*Ts)))//2
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
    fftfreqs = xp.fft.fftshift(xp.fft.fftfreq(X.shape[0], Ts))
    return fftfreqs, X