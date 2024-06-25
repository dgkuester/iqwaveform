"""Some window functions not included by scipy.signal"""

import numpy as np
from scipy import special


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


def taylor(M: int, nbar: int = 4, sll: float = 30, norm=True, sym=True) -> np.ndarray:
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
