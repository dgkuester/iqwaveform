from __future__ import annotations
from .util import lazy_import

import numpy as np
from datetime import datetime
from numbers import Number
import methodtools

signal = lazy_import('scipy.signal')
pylab = lazy_import('matplotlib.pylab')


def correlate_along_axis(a, b, axis=0):
    """cross-correlate `a` and `b` along the specified axis.
    this implementation
    is optimized for small sequences to replace for loop across
    scipy.signal.correlate.
    """
    if axis == 0:
        # np.vdot conjugates b for us
        return np.array([np.vdot(a[:, i], b[:, i]) for i in range(a.shape[1])])
    else:
        return np.array([np.vdot(a[i], b[i]) for i in range(a.shape[0])])


def indexsum2d(ix, iy):
    """take 2 1-D arrays of shape (M,) and (N,) and return a
    2-D array of shape (M,N) with elements (m,n) equal to ix[m,:] + iy[:,n]
    """
    return ix[:, np.newaxis] + iy[np.newaxis, :]


def call_by_block(func, x, size, *args, **kws):
    """repeatedly call `func` on the 1d array `x`, with arguments and keyword arguments args, and kws,
    and concatenate the result
    """
    out_chunks = []
    input_chunks = np.split(x, np.mgrid[: x.size : size][1:])

    if len(input_chunks[-1]) != len(input_chunks[0]):
        input_chunks = input_chunks[:-1]
    for i, chunk in enumerate(input_chunks):
        out_chunks.append(func(chunk, *args, **kws))

    return np.concatenate(out_chunks)


def subsample_shift(x, shift):
    """FFT-based subsample shift in x"""
    N = len(x)
    f = np.fft.fftshift(np.arange(x.size))
    z = np.exp((-2j * np.pi * shift / N) * f)
    return np.fft.ifft(np.fft.fft(x) * z)


def to_blocks(y, size, truncate=False):
    size = int(size)
    if not truncate and y.shape[-1] % size != 0:
        raise ValueError(
            'last axis size {} is not integer multiple of block size {}'.format(
                y.shape[-1], size
            )
        )
    return y[..., : size * (y.shape[-1] // size)].reshape(
        y.shape[:-1] + (y.shape[-1] // size, size)
    )


def _index_or_all(x, name, size):
    if isinstance(x, str) and x == 'all':
        if size is None:
            raise ValueError('must set max to allow "all" value')
        x = np.arange(size)
    elif np.ndim(x) in (0, 1):
        x = np.array(x)
    else:
        raise ValueError(f'{name} argument must be a flat array of indices or "all"')

    if np.max(x) > size:
        raise ValueError(f'{name} value {x} exceeds the maximum {size}')
    if np.max(-x) > size:
        raise ValueError(f'{name} value {x} is below the minimum {-size}')

    return x


class PhyOFDM:
    def __init__(
        self,
        *,
        channel_bandwidth: float,
        sample_rate: float,
        fft_size: float,
        cp_sizes=np.array,
        frame_duration: float | None = None,
        contiguous_size: float | None = None,
    ):
        self.channel_bandwidth = channel_bandwidth
        self.sample_rate = sample_rate

        self.fft_size: float = fft_size
        self.frame_duration = frame_duration

        self.subcarrier_spacing: float = self.sample_rate / fft_size
        if frame_duration is None:
            self.frame_size = None
        else:
            self.frame_size = int(np.rint(sample_rate * frame_duration))

        self.cp_sizes = cp_sizes

        if cp_sizes is None:
            self.contiguous_size = contiguous_size
            self.cp_start_idx = None
            self.cp_idx = None
            self.symbol_idx = None

        else:
            if contiguous_size is not None:
                self.contiguous_size = contiguous_size
            else:
                # if not specified, assume no padding is needed to complete a contiguos block of symbols
                self.contiguous_size = np.sum(cp_sizes) + len(cp_sizes) * fft_size

            # build a (start_idx, size) pair for each CP
            pair_sizes = np.concatenate(((0,), self.cp_sizes + self.fft_size))
            self.cp_start_idx = (pair_sizes.cumsum()).astype(int)[:-1]
            start_and_size = zip(self.cp_start_idx, self.cp_sizes)

            idx_range = range(self.contiguous_size)

            # indices in the contiguous range that are CP
            self.cp_idx = np.concatenate(
                [idx_range[start : start + size] for start, size in start_and_size]
            )

            # indices in the contiguous range that are not CP
            self.symbol_idx = np.array(list(set(idx_range) - set(self.cp_idx)))

    def index_cyclic_prefix(self) -> np.array:
        raise NotImplementedError


class Phy3GPP(PhyOFDM):
    """Sampling and index parameters and lookup tables for 3GPP 5G-NR.

    These are equivalent to LTE if subcarrier_spacing is fixed to 15 kHz
    and slot length is redefined to match the period of 14 symbols including
    cyclic prefix.

    References:
        3GPP TS 38.211.
    """

    # the remaining 1 "slot" worth of samples per slot are for cyclic prefixes
    FFT_PER_SLOT = 14
    SUBFRAMES_PER_PRB = 12

    FFT_SIZE_TO_SUBCARRIERS = {
        128: 73,
        256: 181,
        512: 301,
        1024: 601,
        1536: 901,
        2048: 1201,
    }

    BW_TO_SAMPLE_RATE = {
        1.4e6: 1.92e6,
        3e6: 3.84e6,
        5e6: 7.68e6,
        10e6: 15.36e6,
        15e6: 23.04e6,
        20e6: 30.72e6,
        25e6: 38.40e6,
        30e6: 46.08e6,
        40e6: 61.44e6,
    }

    # Slot structure including cyclic prefix (CP) indices are specified in
    # 3GPP TS 38.211, Section 5.3.1. Below are the sizes of all CPs (in samples)
    # in 1 slot for FFT size 128. CP size then scales proportionally to FFT size.
    # 1 slot is the minimum number of contiguous symbols in a sequence
    MIN_CP_SIZES = np.array((10, 9, 9, 9, 9, 9, 9, 10, 9, 9, 9, 9, 9, 9), dtype=int)

    SCS_TO_SLOTS_PER_FRAME = {15e3: 10, 30e3: 20, 60e3: 40}

    # TODO: add 5G FR2 SCS values
    SUBCARRIER_SPACINGS = {15e3, 30e3, 60e3}

    def __init__(self, channel_bandwidth, subcarrier_spacing=15e3):
        if subcarrier_spacing not in self.SUBCARRIER_SPACINGS:
            raise ValueError(
                f'subcarrier_spacing must be one of {self.SUBCARRIER_SPACINGS}'
            )

        sample_rate = self.BW_TO_SAMPLE_RATE[channel_bandwidth]
        fft_size = int(np.rint(sample_rate / subcarrier_spacing))

        if fft_size in self.FFT_SIZE_TO_SUBCARRIERS:
            self.subcarriers = self.FFT_SIZE_TO_SUBCARRIERS[fft_size]

        super().__init__(
            channel_bandwidth=channel_bandwidth,
            fft_size=fft_size,
            sample_rate=sample_rate,
            frame_duration=10e-3,
            cp_sizes=(fft_size * self.MIN_CP_SIZES) // 128,
        )

    @methodtools.lru_cache(4)
    def index_cyclic_prefix(
        self,
        *,
        frames=(0,),
        symbols='all',
        slots='all',
    ):
        """build an indexing tensor for performing cyclic prefix correlation across various axes"""

        frames = np.array(frames)
        frame_size = int(np.rint(self.sample_rate * 10e-3))

        slots = _index_or_all(
            slots,
            '"slots" argument',
            size=self.SCS_TO_SLOTS_PER_FRAME[self.subcarrier_spacing],
        )
        symbols = _index_or_all(symbols, '"symbols" argument', size=self.FFT_PER_SLOT)

        # first build each grid axis separately
        grid = []

        # axis 0: symbol number within each slot
        grid.append(self.cp_start_idx[symbols])

        # axis 1: slot number
        grid.append(self.contiguous_size * slots)

        # axis 2: frame number
        grid.append(frames * frame_size)

        grid.extend(
            np.ogrid[
                # axis 3: cp index
                0 : self.cp_sizes[1],
                # axis 4: start offset within the symbol
                0 : self.fft_size + self.cp_sizes[1],
            ]
        )

        # pad the axis dimensions so they can be broadcast together
        a = np.meshgrid(*grid, indexing='ij', copy=False)

        # sum all of the index offsets
        inds = a[0].copy()
        for sub in a[1:]:
            inds += sub

        return inds


def isclosetoint(v, atol=1e-6):
    return np.isclose(v % 1, (0, 1), atol=atol).any()


class Phy802_16(PhyOFDM):
    """Sampling and index parameters and lookup tables for IEEE 802.16-2017 OFDMA"""

    VALID_CP_RATIOS = {1 / 32, 1 / 16, 1 / 8, 1 / 4}
    VALID_FFT_SIZES = {128, 512, 1024, 2048}
    VALID_FRAME_DURATIONS = {
        2e-3,
        2.5e-3,
        4e-3,
        5e-3,
        8e-3,
        10e-3,
        12.5e-3,
        20e-3,
        25e-3,
        40e-3,
        50e-3,
    }

    SAMPLING_FACTOR_BY_FREQUENCY_DIV = {
        1.25: 28 / 25,
        1.5: 28 / 25,
        1.75e6: 8 / 7,
        2: 28 / 25,
        2.75: 28 / 25,
    }

    def __init__(
        self,
        channel_bandwidth: float,
        *,
        alt_sample_rate: float = None,
        frame_duration: float = 5e-3,
        fft_size: float = 2048,
        cp_ratio: float = 1 / 8,
    ):
        """
        Args:
            channel_bandwidth: Channel bandwidth as defined by 802.16-2017
            alt_sample_rate (_type_, optional): If specified, overrides the 802.16-2017 value with sample rate of recorded data.
            frame_duration: _description_. Defaults to 5e-3.
            fft_size: the fft size corresponding to the length of the useful portion of each symbol. Defaults to 2048.
            cp_ratio: the size of the cyclic prefix as a fraction of fft_size. Defaults to 1/8.
        """
        if not isinstance(channel_bandwidth, Number):
            raise TypeError('expected numeric value for channel_bandwidth')
        elif channel_bandwidth < 1.25e6:
            raise ValueError(
                'standardized values for channel_bandwidth not supported yet'
            )
        elif not np.isclose(channel_bandwidth % 125e3, 0, atol=1e-6):
            raise ValueError('channel bandwidth must be set in increments of 125 kHz')

        if fft_size not in self.VALID_FFT_SIZES:
            raise ValueError(f'fft_size must be one of {self.VALID_FFT_SIZES}')

        if cp_ratio in self.VALID_CP_RATIOS:
            self.cp_ratio = cp_ratio
        else:
            raise ValueError(f'cp_ratio must be one of {self.VALID_CP_RATIOS}')

        if frame_duration not in self.VALID_FRAME_DURATIONS:
            raise ValueError(
                f'frame_duration must be one of {self.VALID_FRAME_DURATIONS}'
            )

        for freq_divisor, n in self.SAMPLING_FACTOR_BY_FREQUENCY_DIV.items():
            if np.isclose(channel_bandwidth % freq_divisor, 0, atol=1e-6):
                sampling_factor = self.sampling_factor = n
                break
        else:
            # no match with the table - standardized default
            sampling_factor = self.sampling_factor = 8 / 7

        std_sample_rate = np.floor(sampling_factor * channel_bandwidth / 8000) * 8000
        cp_size = int(np.rint(cp_ratio * fft_size))
        self.total_symbol_duration = (
            int(np.rint((1 + cp_ratio) * fft_size)) / std_sample_rate
        )
        self.symbols_per_frame = int(
            np.floor(frame_duration / self.total_symbol_duration)
        )

        if alt_sample_rate is None:
            sample_rate = std_sample_rate
        else:
            scale = alt_sample_rate / std_sample_rate

            if not (isclosetoint(scale) or isclosetoint(1 / scale)):
                raise ValueError(
                    'alt_sample_rate must be integer multiple or divisor of ofdm sample_rate'
                )
            if not isclosetoint(cp_size * scale):
                raise ValueError(
                    'alt_sample_rate is too small to capture any cyclic prefixes'
                )

            fft_size = int(np.rint(fft_size * scale))
            cp_size = int(np.rint(cp_size * scale))
            sample_rate = alt_sample_rate

        # we need to specify explicitly the length of the contiguous sequence
        # of symbols to ensure proper padding out to a full frame
        contiguous_size = int(np.rint(frame_duration * sample_rate))

        super().__init__(
            channel_bandwidth=channel_bandwidth,
            fft_size=fft_size,
            sample_rate=sample_rate,
            frame_duration=frame_duration,
            cp_sizes=np.full(self.symbols_per_frame, cp_size),
            contiguous_size=contiguous_size,
        )

    @methodtools.lru_cache(4)
    def index_cyclic_prefix(
        self,
        *,
        frames=(0,),
        symbols='all',
    ) -> np.ndarray:
        """build an indexing tensor for performing cyclic prefix correlation across various axes"""

        frames = np.array(frames)

        symbols = _index_or_all(
            symbols, '"symbols" argument', size=self.symbols_per_frame
        )

        # first build each grid axis separately
        grid = []

        # axis 0: symbol number in each frame
        grid.append(self.cp_start_idx[symbols])

        # axis 1: frame number
        grid.append(frames * self.frame_size)

        grid.extend(
            np.ogrid[
                # axis 2: cp index
                0 : self.cp_sizes[1],
                # axis 3: start offset within the symbol
                0 : self.fft_size + self.cp_sizes[1],
            ]
        )

        # pad the axis dimensions so they can be broadcast together
        a = np.meshgrid(*grid, indexing='ij', copy=False)

        # sum all of the index offsets
        inds = a[0].copy()
        for sub in a[1:]:
            inds += sub

        return inds


empty_complex64 = np.zeros(0, dtype=np.complex64)


class BasebandClockSynchronizer:  # other base classes are basic_block, decim_block, interp_block
    """Use the cyclic prefix (CP) in the LTE PHY layer to
    (1) resample to correct clock mismatch relative to the transmitter, and
    (2) align LTE signal to the start of a CP (resolution of fft_size*(1+9/128) samples)

    Usage:

        sync = BasebandClockSynchronizer(channel_bandwidth=channel_bandwidth)
        y = sync(x, 0.1)

    Notes:

        This has not been optimized for low SNR, so it it best used in laboratory settings
        with strong signal.

    """

    COARSE_CP0_STEP = (
        1.0 / 6
    )  # coarse search step, as fraction of the length of the first cyclic prefix

    def __init__(
        self,
        channel_bandwidth: float,  # the channel bandwidth in Hz: 1.4 MHz, 5 MHz, 10 MHz, 20 MHz, etc
        correlation_subframes: int = 20,  # correlation window size, in subframes (20 = 1 frame)
        sync_window_count: int = 2,  # how many correlation windows to synchronize at a time (suggest >= 2)
        which_cp: str = 'all',  # 'all', 'special', or 'normal'
        subcarrier_spacing=15e3,
    ):
        self.phy = Phy3GPP(channel_bandwidth, subcarrier_spacing=subcarrier_spacing)
        self.correlation_subframes = correlation_subframes
        self.sync_size = (
            sync_window_count * correlation_subframes * self.phy.contiguous_size
        )

        # index array of cyclic prefix samples
        cp_gate = self.phy.cp_idx  # 1 single slot
        i_slot_starts = self.phy.contiguous_size * np.arange(correlation_subframes)
        cp_gate = indexsum2d(
            i_slot_starts, cp_gate
        ).flatten()  # duplicate across slot_count slots

        # Define a coarse sample grid to get into the ballpark of the correlation peak.
        # self.COARSE_CP0_STEP defines the resolution of the search, in fractions of the
        # standard size (i.e., not the first) CP window.
        #
        # This grid spans a total length of a single slot.
        coarse_step = int(self.phy.cp_sizes[1] * self.COARSE_CP0_STEP)
        self.cp_offsets_coarse = np.arange(
            0, self.phy.fft_size + self.phy.cp_sizes[1], coarse_step, dtype=int
        )

        # 2-D search grid
        self.cp_indices_coarse = indexsum2d(self.cp_offsets_coarse, cp_gate)

        # Define the fine search sample grid, which is applied as an offset relative to the
        # result of the coarse search
        self.cp_offsets_fine = np.arange(
            -np.ceil(coarse_step / 2), np.ceil(coarse_step / 2) + 1, 1, dtype=int
        )
        self.cp_indices_fine = indexsum2d(self.cp_offsets_fine, cp_gate)

    def _cp_correlate(self, x, cp_inds):
        """
        :cp_inds: 2D index offsets array of shape (M,N), where:
                  dimension 0 indices are trial index locations for the start of the slot
                  dimension 1 gives corresponding index offsets of CP samples relative to the start of the slot

        """
        return correlate_along_axis(x[cp_inds], x[self.phy.fft_size :][cp_inds], axis=1)

    def _find_slot_start_offset(self, x):
        """Estimate the offset required to align the start of a slot to
        index 0 in the complex sample vector `x`
        """
        self._debug.setdefault('x', []).append(x)

        # Coarse estimate of alignment offset to within coarse_step samples
        coarse_corr = np.abs(self._cp_correlate(x, self.cp_indices_coarse))
        self._debug.setdefault('coarse_corr', []).append(coarse_corr)
        coarse_offset = self.cp_offsets_coarse[np.argmax(coarse_corr)]

        # Fine calculation for the offset (near the coarse result)
        fine_corr = np.abs(self._cp_correlate(x, self.cp_indices_fine + coarse_offset))
        n_fine = np.argmax(fine_corr)
        fine_offset = coarse_offset + self.cp_offsets_fine[n_fine]

        noise_est = np.nanmedian(np.abs(np.sort(coarse_corr)[:-3]))

        return fine_offset, fine_corr[n_fine], noise_est

    def _offset_by_sync_period(self, x):
        """Given the LTE baseband signal in complex array `x`, enforce synchronization
        to the start of a slot every `sync_size` samples.

        To find the offset required in each sync period, cyclic prefix correlation is
        applied to the full window between sync periods.
        """

        self._debug = {}

        ret = []
        input_chunks = np.split(x, np.mgrid[: x.size : self.sync_size][1:])

        if len(input_chunks[-1]) != len(input_chunks[0]):
            input_chunks = input_chunks[:-1]

        ret = [self._find_slot_start_offset(chunk) for chunk in input_chunks]

        return np.array(ret)

    def _estimate_clock_mismatch(self, x, snr_min=3):
        """Phase-unwrapped linear regression to estimate the discrepancy
        between the transmit and receive baseband clock frequencies across
        all synchronization windows.
        """

        from sklearn import linear_model

        offsets, weights, noise = self._offset_by_sync_period(x).T
        t_sync = (self.sync_size / self.phy.sample_rate) * np.arange(offsets.size)

        self.snr = weights / noise

        # Require a minimum SNR to be included in the estimate. (otherwise,
        # noisy values are a potential problem for np.unwrap - this is still possibly
        # a lingering problem, even with this in place)
        select = self.snr > snr_min  # "good" indices

        print(
            f'  {select.sum()} sync windows had well-correlated cyclic prefix ({select.sum() / select.size * 100:0.1f}%)'
        )
        offsets = offsets[select]
        t_sync = t_sync[select]
        weights = weights[select]

        # the offsets wrap back to zero when they pass fft_size+length of the first CP block.
        # unwrap here to keep linear regression from breaking
        offsets = self._unwrap_offsets(offsets)

        print('LinearRegression.fit()')
        print('tsync shape:' + str(t_sync.shape))
        print('offsets shape:' + str(offsets.shape))
        print('weights shape:' + str(weights.shape))
        fit = linear_model.LinearRegression().fit(
            t_sync.reshape(-1, 1), offsets.reshape(-1, 1), weights
        )
        print('LinearRegression.fit() finished')
        slipped_samples = np.round(
            fit.coef_[0, 0] * x.size / self.phy.sample_rate
        ).astype(int)

        self._regression_info = dict(
            inputs=(t_sync, offsets, weights), fit=fit, slipped_samples=slipped_samples
        )

        return slipped_samples, fit.intercept_[0]

    def _unwrap_offsets(self, offsets):
        scale_rad = 2 * np.pi / self.phy.fft_size
        return (np.unwrap(offsets * scale_rad) / scale_rad).astype(int)

    def plot_offset_with_fit(self, x):
        slope, intercept = self._estimate_clock_mismatch(x)
        t, offsets, weights = self._regression_info['inputs']
        pylab.plot(t, offsets, '.')
        pylab.plot(
            t, t * self._regression_info['slope'] + self._regression_info['intercept']
        )

    def __call__(
        self, x, subsample_offset_correction=True, max_passes=10, on_fail='except'
    ):
        """Resample to correct for baseband clock mismatch.

        :subsample_offset_correction:
            * True to perform subsampling
            * False round to the nearest clock offset for speed
        """

        # if there are enough sample slips, we might slip across 1 or more CP windows.
        # biasing the estimate of the number of slipped samples. work through this iteratively by successive
        # attempts at resampling.
        total_sample_slip = 0
        for i in range(max_passes + 1):
            print(f'baseband clock correction pass {i + 1}')
            sample_slip, offset = self._estimate_clock_mismatch(x)
            total_sample_slip += sample_slip

            if sample_slip == 0:
                break
            else:
                # resample to correct sample slipping
                print('resampling')
                print('sample slip iiiiiiis: ' + str(sample_slip))
                print('total samples is: ' + str(x.size - sample_slip))
                #                print('forcing sample slip to be 8.4Hz')
                #                sample_slip = 42
                now = datetime.now()
                print(
                    'start resample with sample_slip: '
                    + str(sample_slip)
                    + ' '
                    + str(now)
                )
                x = signal.resample(x, x.size - sample_slip)
                elapsed = datetime.now() - now
                print('done resampling ' + str(sample_slip) + ' ' + str(elapsed))

        else:
            if on_fail == 'except':
                raise ValueError(
                    f'failed to converge on clock mismatch within {max_passes} passes'
                )

        print(
            f'corrected baseband clock slip by {total_sample_slip} samples'
            f'({total_sample_slip / x.size * self.phy.sample_rate:0.2f} Hz clock mismatch)'
        )

        # last, correct the fixed offset at the beginning in an attempt to align
        if subsample_offset_correction:
            print(f'subsample shift to correct offset of {offset:0.3f} samples')
            x = subsample_shift(x, -offset)
        else:
            int_offset = int(offset.round())
            print(
                f'shift to correct offset of {int_offset} (out of {offset:0.3f}) samples'
            )
            x = x[int_offset % self.phy.contiguous_size :]

            # keep only an integer number of slots
        spare_samples = x.size % (2 * self.phy.contiguous_size)
        if spare_samples > 0:
            x = x[:-spare_samples]

        # if there are enough sample slips, we might slip backward by 1 CP duration (9/128 * self.phy.fft_size)
        return x


class SymbolDecoder:
    """Decode symbols from a clock-synchronized received waveform. This uses simple LTE PHY numerology,
    and an edge detection scheme to synchronize symbols relative to slots.

    Usage:

        decode = SymbolDecoder(channel_bandwidth=channel_bandwidth)
        y = decode(x)

    Notes:
        This has not been optimized for low SNR, so it it best used in laboratory settings
        with strong signal.

    """

    def __init__(self, channel_bandwidth):
        self.phy = Phy3GPP(channel_bandwidth)

    @staticmethod
    def prb_power(symbols):
        """Return the total power in the PRB"""
        return (np.abs(to_blocks(symbols, Phy3GPP.SUBFRAMES_PER_PRB)) ** 2).sum(axis=-1)

    def _decode_symbols(self, x, only_3gpp_subcarriers=True):
        # first, select symbol indices (== remove cyclic prefixes)
        x = to_blocks(x, 2 * self.phy.contiguous_size)[:, self.phy.symbol_idx].flatten()

        # break up the waveform into windows of length fft_size
        blocks = to_blocks(x, self.phy.fft_size)

        #  decode with the fft
        X = np.fft.fftshift(np.fft.fft(blocks, axis=-1), axes=(-1,))

        X /= np.sqrt(2 * self.phy.fft_size)

        if only_3gpp_subcarriers:
            # return only the FFT bins meant to contain data
            sc_start = X.shape[-1] // 2 - self.phy.subcarriers // 2
            sc_stop = X.shape[-1] // 2 + self.phy.subcarriers // 2
            X = X[:, sc_start:sc_stop]
        print(x.shape)
        return X

    def _align_symbols_to_tti(self, symbols):
        # determine the power change that is strongest across all PRBs in each FFT window
        power = self.prb_power(symbols)
        power_diff = np.diff(power, axis=0, append=0) / power
        diff_peaks = np.abs(power_diff).max(axis=1)
        diff_peak_by_symbol = to_blocks(diff_peaks, Phy3GPP.FFT_PER_SLOT)
        self._diff_peak_by_symbol = diff_peak_by_symbol
        self._diff_peaks = diff_peaks
        self._power_diff = power_diff

        # where do the maxima occur in each tti
        tti_offset = diff_peak_by_symbol.max(axis=0).argmax() + 1

        return symbols[tti_offset:]

    def __call__(self, x):
        """Ringlead the decoding process"""
        symbols = self._decode_symbols(x)
        symbols = self._align_symbols_to_tti(symbols)
        return symbols
