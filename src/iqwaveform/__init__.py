import os

os.environ.setdefault('NUMEXPR_MAX_THREADS', '4')
del os

from ._version import __version__

from . import fourier, io, ofdm, power_analysis, util

figures = util.lazy_import('iqwaveform.figures')

from .fourier import to_blocks, iqfftfreq, stft, iq_to_stft_spectrogram

from .figures import (
    pcolormesh_df,
    plot_power_histogram_heatmap,
    plot_power_ccdf,
    plot_spectrogram_heatmap_from_iq,
    plot_spectrogram_heatmap,
)

from .io import waveform_to_frame

from .power_analysis import (
    dBtopow,
    envtodB,
    envtopow,
    hist_laxis,
    iq_to_bin_power,
    iq_to_cyclic_power,
    power_histogram_along_axis,
    powtodB,
    sample_ccdf,
)

from .util import isroundmod
