import os

os.environ.setdefault('NUMEXPR_MAX_THREADS', '4')
del os

from ._version import __version__

from . import fourier, io, ofdm, power_analysis, util

figures = util.lazy_import('iqwaveform.figures')

from .power_analysis import (
    dBtopow,
    powtodB,
    envtodB,
    envtopow,
    hist_laxis,
    power_histogram_along_axis,
    sample_ccdf,
    iq_to_bin_power,
)

from .fourier import to_blocks, stft, iq_to_stft_spectrogram

from .figures import (
    pcolormesh_df,
    plot_power_histogram_heatmap,
    plot_power_ccdf,
    plot_spectrogram_heatmap_from_iq,
    plot_spectrogram_heatmap,
)

from .io import waveform_to_frame
