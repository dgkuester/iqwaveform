from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import signal, special
from . import power_analysis
import scipy
from multiprocessing import cpu_count
from functools import partial, lru_cache
from array_api_compat import array_namespace
from array_api_strict._typing import Array


CPU_COUNT = cpu_count()

fft = partial(scipy.fft.fft, workers=CPU_COUNT // 2, overwrite_x=True)
ifft = partial(scipy.fft.ifft, workers=CPU_COUNT // 2, overwrite_x=True)


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
        w = xp.array(_get_window(name_or_tuple, N))
        if dtype is not None:
            w = w.astype(dtype)
        return w


def broadcast_onto(a: Array, other: Array, axis: int) -> Array:
    """broadcast a 1-D array onto a specified axis of `other`"""
    xp = array_namespace(a, other)

    slices = [xp.newaxis] * len(other.shape)
    slices[axis] = slice(None, None)
    return a.__getitem__(tuple(slices))


@lru_cache(16)
def _get_stft_axes(fs: float, fft_size: int, time_size: int, overlap_frac: float=0, xp=np) -> tuple[Array, Array]:
    """returns stft (freqs, times) array tuple appropriate to the array module xp"""

    freqs = xp.fft.fftshift(xp.fft.fftfreq(fft_size, d=1/fs))
    times = xp.arange(time_size) * ((1-overlap_frac) * fft_size / fs)

    return freqs, times


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
):
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

    # # This is probably the same
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
    if noverlap not in (0, fft_size // 2):
        raise NotImplementedError('noverlap must be noverlap//2 or 0')

    if norm not in ('power', None):
        raise TypeError('norm must be "power" or None')

    if isinstance(window, str) or (isinstance(window, tuple) and len(window) == 2):
        should_norm = (norm == 'power')
        w = _get_window(
            window, fft_size, xp=xp, dtype=x.dtype, norm=should_norm
        )
    else:
        w = xp.array(window)

    if noverlap == 0:
        x = to_blocks(x, fft_size, truncate=truncate)
        X = xp.fft.fftshift(
            fft(
                x * broadcast_onto(w / fft_size, x, 1),
                axis=axis + 1
            ),
            axes=axis + 1,
        )

    else:
        if axis != 0:
            raise NotImplementedError(
                'for now, only axis=0 is supported with noverlap>0'
            )

        x = xp.array([x[:-noverlap], x[noverlap:]])
        x = to_blocks(x, fft_size, axis=1, truncate=truncate)

        # X = xp.empty((x.shape[0], 2, FFT_SIZE) + x.shape[2:])

        x *= broadcast_onto(w / fft_size, x, 2)
        X = xp.fft.fft(x, axis=axis + 2, workers=CPU_COUNT // 2, overwrite_x=True)

        # interleave the 2 overlapping offsets, and axis shift
        shape = (X.shape[0] * X.shape[1],) + X.shape[2:]
        X = xp.swapaxes(X, 0, 1).reshape(shape)
        X = xp.fft.fftshift(X, axes=axis + 1)

    freqs, times = _get_stft_axes(
        fs,
        fft_size=fft_size,
        time_size=X.shape[axis],
        overlap_frac=noverlap/fft_size,
        xp=xp
    )

    return freqs, times, X


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

    # print((xp.abs(fftfreqs)>=bandwidth/2).sum(), (xp.abs(fftfreqs)<bandwidth/2).sum())

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


def iq_to_stft_spectrogram(iq: Array, window: Array | str | tuple[str,float],
                           fft_size: int, Ts, overlap=True, analysis_bandwidth=None):
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


def _len_guards(M):
    """Handle small or incorrect window lengths"""
    if int(M) != M or M < 0:
        raise ValueError('Window length M must be a non-negative integer')
    return M <= 1


def _extend(M, sym):
    """Extend window by 1 sample if needed for DFT-even symmetry"""
    if not sym:
        return M + 1, True
    else:
        return M, False


def _truncate(w, needed):
    """Truncate window by 1 sample if needed for DFT-even symmetry"""
    if needed:
        return w[:-1]
    else:
        return w


def taylor(M: int, nbar: int = 4, sll:float = 30, norm=True, sym=True) -> np.ndarray:
    """
    Return a Taylor window.
    The Taylor window taper function approximates the Dolph-Chebyshev window's
    constant sidelobe level for a parameterized number of near-in sidelobes,
    but then allows a taper beyond [2]_.
    The SAR (synthetic aperature radar) community commonly uses Taylor
    weighting for image formation processing because it provides strong,
    selectable sidelobe suppression with minimum broadening of the
    mainlobe [1]_.
    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.
    nbar : int, optional
        Number of nearly constant level sidelobes adjacent to the mainlobe.
    sll : float, optional
        Desired suppression of sidelobe level in decibels (dB) relative to the
        DC gain of the mainlobe. This should be a positive number.
    norm : bool, optional
        When True (default), divides the window by the largest (middle) value
        for odd-length windows or the value that would occur between the two
        repeated middle values for even-length windows such that all values
        are less than or equal to 1. When False the DC gain will remain at 1
        (0 dB) and the sidelobes will be `sll` dB down.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.
    Returns
    -------
    out : array
        The window. When `norm` is True (default), the maximum value is
        normalized to 1 (though the value 1 does not appear if `M` is
        even and `sym` is True).
    See Also
    --------
    chebwin, kaiser, bartlett, blackman, hamming, hanning
    References
    ----------
    .. [1] W. Carrara, R. Goodman, and R. Majewski, "Spotlight Synthetic
           Aperture Radar: Signal Processing Algorithms" Pages 512-513,
           July 1995.
    .. [2] Armin Doerry, "Catalog of Window Taper Functions for
           Sidelobe Control", 2017.
           https://www.researchgate.net/profile/Armin_Doerry/publication/316281181_Catalog_of_Window_Taper_Functions_for_Sidelobe_Control/links/58f92cb2a6fdccb121c9d54d/Catalog-of-Window-Taper-Functions-for-Sidelobe-Control.pdf
    Examples
    --------
    Plot the window and its frequency response:
    >>> from scipy import signal
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt
    >>> window = signal.windows.taylor(51, nbar=20, sll=100, norm=False)
    >>> plt.plot(window)
    >>> plt.title('Taylor window (100 dB)')
    >>> plt.ylabel('Amplitude')
    >>> plt.xlabel('Sample')
    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window) / 2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title('Frequency response of the Taylor window (100 dB)')
    >>> plt.ylabel('Normalized magnitude [dB]')
    >>> plt.xlabel('Normalized frequency [cycles per sample]')
    """  # noqa: E501

    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    # Original text uses a negative sidelobe level parameter and then negates
    # it in the calculation of B. To keep consistent with other methods we
    # assume the sidelobe level parameter to be positive.
    B = 10 ** (sll / 20)
    A = np.arccosh(B) / np.pi
    s2 = nbar**2 / (A**2 + (nbar - 0.5) ** 2)
    ma = np.arange(1, nbar)

    Fm = np.empty(nbar - 1)
    signs = np.empty_like(ma)
    signs[::2] = 1
    signs[1::2] = -1
    m2 = ma * ma
    for mi, m in enumerate(ma):
        numer = signs[mi] * np.prod(1 - m2[mi] / s2 / (A**2 + (ma - 0.5) ** 2))
        denom = 2 * np.prod(1 - m2[mi] / m2[:mi]) * np.prod(1 - m2[mi] / m2[mi + 1 :])
        Fm[mi] = numer / denom

    def W(n):
        return 1 + 2 * np.dot(
            Fm, np.cos(2 * np.pi * ma[:, np.newaxis] * (n - M / 2.0 + 0.5) / M)
        )

    w = W(np.arange(M))

    # normalize (Note that this is not described in the original text [1])
    if norm:
        scale = 1.0 / W((M - 1) / 2)
        w *= scale

    return _truncate(w, needs_trunc)


def knab(M: int, alpha, sym=True) -> np.ndarray:
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    t = np.linspace(-0.5, 0.5, M)

    sqrt_term = np.sqrt(1 - (2 * t) ** 2)
    w = np.sinh((np.pi * alpha) * sqrt_term) / (np.sinh(np.pi * alpha) * sqrt_term)

    w[0] = w[-1] = np.pi * alpha / np.sinh(np.pi * alpha)
    w /= np.sqrt(np.sum(w**2))

    return _truncate(w, needs_trunc)


def modified_bessel(M, alpha, sym=True):
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    t = np.linspace(-0.5, 0.5, M)

    sqrt_term = np.sqrt(1 - (2 * t) ** 2)
    w = special.i1((np.pi * alpha) * sqrt_term) / (
        special.i1(np.pi * alpha) * sqrt_term
    )

    w[0] = w[-1] = 0  # np.pi*alpha/np.sinh(np.pi*alpha)

    w /= np.sqrt(np.sum(w**2))

    return _truncate(w, needs_trunc)


def cosh(M: int, alpha, sym=True) -> np.ndarray:
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    t = np.linspace(-0.5, 0.5, M)

    sqrt_term = np.sqrt(1 - (2 * t) ** 2)
    w = np.cosh((np.pi * alpha) * sqrt_term) / (np.cosh(np.pi * alpha) * sqrt_term)

    w[0] = w[-1] = 1 / np.cosh(np.pi * alpha)

    w /= np.sqrt(np.sum(w**2))

    return _truncate(w, needs_trunc)


# def modified_bessel(M: int, alpha, sym=True) -> np.ndarray:
#     if _len_guards(M):
#         return np.ones(M)
#     M, needs_trunc = _extend(M, sym)

#     t = np.linspace(-0.5, 0.5, M)

#     sqrt_term = np.sqrt(1 - (2 * t) ** 2)
#     w = special.i1((np.pi * alpha) * sqrt_term) / (
#         special.i1(np.pi * alpha) * sqrt_term
#     )

#     w[0] = w[-1] = 0  # np.pi*alpha/np.sinh(np.pi*alpha)

#     w /= np.sqrt(np.sum(w**2))

#     return _truncate(w, needs_trunc)
