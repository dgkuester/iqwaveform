import numba as nb
import math

@nb.njit(
    # [
    #     (nb.int32[:], nb.complex64[:], nb.int32, nb.int32, nb.boolean, nb.complex64[:]),
    #     (nb.int32[:], nb.complex64[:], nb.int64, nb.int32, nb.boolean, nb.complex64[:]),
    #     (nb.int64[:], nb.complex64[:], nb.int32, nb.int32, nb.boolean, nb.complex64[:]),
    #     (nb.int64[:], nb.complex64[:], nb.int64, nb.int32, nb.boolean, nb.complex64[:]),
    #     (nb.int32[:], nb.complex64[:], nb.int32, nb.int64, nb.boolean, nb.complex64[:]),
    #     (nb.int32[:], nb.complex64[:], nb.int64, nb.int64, nb.boolean, nb.complex64[:]),
    #     (nb.int64[:], nb.complex64[:], nb.int32, nb.int64, nb.boolean, nb.complex64[:]),
    #     (nb.int64[:], nb.complex64[:], nb.int64, nb.int64, nb.boolean, nb.complex64[:]),
    # ],
    parallel=True,
    cache=True
)
def _corr_at_indices(inds, x, nfft: int, ncp: int, norm: bool, out):
    # j: autocorrelation sequence (output) index
    for j in nb.prange(nfft + ncp):
        accum_corr = nb.complex128(0 + 0j)
        accum_power_a = nb.float64(0.0)
        accum_power_b = nb.float64(0.0)

        # i: the sample index of each waveform sample to compare against its cyclic shift
        for i in range(inds.shape[0]):
            ix = inds[i] + j

            if ix > x.shape[0]:
                break

            a = x[ix]
            b = x[ix + nfft]
            bconj = b.conjugate()
            accum_corr += a * bconj
            if norm:
                accum_power_a += (a * a.conjugate()).real
                accum_power_b += (b * bconj).real

        if norm:
            # normalize by the standard deviation under the assumption
            # that the voltage has a mean of zero
            accum_corr /= math.sqrt(accum_power_a * accum_power_b)
        else:
            # power normalization: scale by number of indices
            accum_corr /= inds.shape[0]

        out[j] = accum_corr
