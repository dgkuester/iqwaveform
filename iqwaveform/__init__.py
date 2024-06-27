import os

os.environ.setdefault('NUMEXPR_MAX_THREADS', '4')
del os

from ._version import __version__

from . import ofdm, fourier, figures, io, power_analysis

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

# the following is intended to accommodate matplotlib 3.6. it is not
# necessary for >= 3.7, and may be removed in the future
import matplotlib as mpl
from pathlib import Path

styles = {
    __name__ + '.' + sfile.stem: mpl.rc_params_from_file(
        sfile, use_default_template=False
    )
    for sfile in Path(__file__).parent.glob('*.mplstyle')
}
mpl.style.library.update(styles)
del mpl, Path, styles
