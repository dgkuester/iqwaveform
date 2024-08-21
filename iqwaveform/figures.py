from __future__ import annotations
import math
import numpy as np

from .power_analysis import powtodB, dBtopow, envtodB, sample_ccdf, iq_to_bin_power
from .fourier import iq_to_stft_spectrogram
from . import type_stubs
from .util import lazy_import

mpl = lazy_import('matplotlib')
stats = lazy_import('scipy.stats')
pd = lazy_import('pandas')


def _show_xarray_units_in_parentheses():
    from xarray.plot.utils import _get_units_from_attrs

    code = _get_units_from_attrs.__code__
    consts = tuple([' ({})' if c == ' [{}]' else c for c in code.co_consts])
    _get_units_from_attrs.__code__ = code.replace(co_consts=consts)


_show_xarray_units_in_parentheses()


def is_decade(x, **kwargs):
    y = np.log10(x)
    return np.isclose(y, np.round(y), **kwargs)


class GammaMaxNLocator(mpl.ticker.MaxNLocator):
    """The ticker locator for linearized gamma-distributed survival functions"""

    def __init__(self, transform, nbins=None, minor=False):
        self._transform = transform
        self._minor = minor
        super().__init__(nbins)

    def __call__(self):
        dmin, dmax = self.axis.get_data_interval()
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(max(vmin, dmin), min(vmax, dmax))

    def tick_values(self, vmin, vmax):
        vmin, vmax = min((vmin, vmax)), max((vmin, vmax))
        vmin, vmax = self.limit_range_for_scale(vmin, vmax, 1e-7)
        vth_lo = 0.15
        vth_hi = 0.75

        # compute bounds in the transformed space, where step sizes will be roughly even
        tmin = self._transform.transform(vmin)
        tmax = self._transform.transform(vmax)
        tth_lo = self._transform.transform(vth_lo)
        tth_hi = self._transform.transform(vth_hi)

        # the lo and hi regions will be logarithmically spaced; the middle linear
        counts_lo = int(max(0, np.ceil(self._nbins * (tth_lo - tmin) / (tmax - tmin))))
        counts_hi = int(max(0, np.ceil(self._nbins * (tmax - tth_hi) / (tmax - tmin))))
        counts_mid = self._nbins - (counts_lo + counts_hi)

        # now compute round tick locations in the original space (0 to 1)
        ticks = []
        if counts_lo > 0:
            if self._minor:
                lo_locator = mpl.ticker.LogLocator(
                    base=10.0, subs=(1.0, 3), numticks=counts_lo
                )
                ticks.append(lo_locator.tick_values(vmin, 0.1))
            else:
                ticks.append(np.logspace(-(counts_lo), -1, counts_lo))

        if counts_mid > 0 and not self._minor:
            mid_locator = mpl.ticker.MaxNLocator(
                nbins=int(np.ceil(counts_mid / 2)), steps=[1, 2, 2.5, 5, 10]
            )
            ticks.append(mid_locator.tick_values(vth_lo, 0.5))
            ticks.append(mid_locator.tick_values(0.5, vth_hi))

        if counts_hi > 0:
            if self._minor:
                hi_locator = mpl.ticker.LogLocator(
                    base=10.0, subs=(1.0, 2, 3, 5), numticks=counts_hi
                )
                ticks.append(1 - hi_locator.tick_values(1 - 0.9, 1 - vmax))
            else:
                ticks.append([0.9])
                if counts_hi > 1:
                    ticks.append([0.95])
                if counts_hi > 2:
                    ticks.append(1 - np.logspace(-1, -(counts_hi - 2), counts_hi - 2))

        ticks = np.concatenate(ticks)
        ticks = np.unique(ticks)
        if 0.0 in ticks:
            ticks = ticks[1:]

        if self._minor:
            ticks = ticks[(ticks < 0.2) | (ticks > 0.85)]

        return ticks

    def get_transform(self):
        return self._transform

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """Limit the domain to positive values."""
        vmin, vmax = min((vmin, vmax)), max((vmin, vmax))

        if not np.isfinite(minpos):
            minpos = 1e-12  # Should rarely (if ever) have a visible effect.

        ret = (
            minpos if vmin <= minpos else vmin,
            1.0 - np.sqrt(minpos) if vmax >= 1 - np.sqrt(minpos) else vmax,
        )

        self.axis.set_view_interval(ret[1], ret[0], True)
        # self.axis.set_data_interval(ret[1], ret[0], True)

        return ret

    def view_limits(self, vmin, vmax):
        vmin, vmax = self.nonsingular(vmin, vmax)
        return vmin, vmax


class GammaLogitFormatter(mpl.ticker.LogitFormatter):
    """A text formatter for probability labels on the GammaCCDF scale"""

    def __call__(self, x, pos=None):
        if self._minor and x not in self._labelled:
            return ''
        if x <= 0 or x >= 1:
            return ''
        if math.isclose(2 * x, round(2 * x)) and round(2 * x) == 1:
            s = self._one_half
        elif np.any(np.isclose(x, np.array([0.01, 0.1, 0.9, 0.99]), rtol=1e-7)):
            s = str(x)
        elif x < 0.1 and is_decade(x, rtol=1e-7):
            exponent = round(np.log10(x))
            s = '10^{%d}' % exponent
        elif x > 0.9 and is_decade(1 - x, rtol=1e-7):
            exponent = round(np.log10(1 - x))
            s = self._one_minus('10^{%d}' % exponent)
        elif x < 0.05:
            s = self._format_value(x, self.locs)
        elif x > 0.98:
            s = self._one_minus(self._format_value(1 - x, 1 - self.locs))
        else:
            s = self._format_value(x, self.locs, sci_notation=False)
        return r'$\mathdefault{%s}$' % s


class GammaQQScale(mpl.scale.FuncScale):
    """A transformed scale that linearizes Gamma-distributed survival functions when the
    independent axis is log-scaled (e.g., dB).

    Suggested usage:

    ```
        from iqwaveform import figures

        plot(10*np.log10(bins), sf)

        ax.set_scale('gamma-ccdf', k=10)

    ```
    In power measurements, the shape parameter `k` should be set equal to the number of averaged power samples.

    """

    name = 'gamma-qq'

    def __init__(
        self,
        axis,
        *,
        k,
        major_ticks=10,
        minor_ticks=None,
        vmin=None,
        vmax=None,
        db_ordinal=True,
    ):
        def forward(q):
            x = stats.gamma.isf(q, a=k, scale=1)
            if db_ordinal:
                x = powtodB(x)
            return x

        def inverse(x):
            if db_ordinal:
                x = dBtopow(x)
            q = 1 - stats.gamma.sf(x, a=k, scale=1)
            return q

        transform = mpl.scale.FuncTransform(forward=forward, inverse=inverse)
        self._major_locator = GammaMaxNLocator(transform=transform, nbins=major_ticks)

        if minor_ticks is not None:
            self._minor_locator = GammaMaxNLocator(
                transform=transform, nbins=minor_ticks, minor=True
            )
        else:
            self._minor_locator = None

        super().__init__(axis, (forward, inverse))

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(self._major_locator)

        if self._minor_locator is not None:
            axis.set_minor_locator(self._minor_locator)

        axis.set_major_formatter(GammaLogitFormatter(one_half='0.5'))


mpl.scale.register_scale(GammaQQScale)


def contiguous_segments(df, index_level, threshold=7, relative=True):
    """Split `df` into a list of DataFrames for which the index values
    labeled by level `index_level`, have no discontinuities greater
    than threshold*(median step between index values).
    """
    delta = pd.Series(df.index.get_level_values(index_level)).diff()
    if relative:
        threshold = threshold * delta.median()
    i_gaps = delta[delta > threshold].index.values
    i_segments = [[0] + list(i_gaps), list(i_gaps) + [None]]

    return [df.iloc[i0:i1] for i0, i1 in zip(*i_segments)]


def _has_tick_label_collision(ax, which: str, spacing_threshold=10):
    """finds the minimum spacing between tick labels along an axis to check for collisions (overlaps).

    Args:
        ax: matplotlib the axis object

        which: "x" or "y"

    Returns:
        the spacing, in units of the figure render of the axis. negative indicates a collision
    """
    fig = ax.get_figure()

    if which == 'x':
        the_ax = ax.xaxis
    elif which == 'y':
        the_ax = ax.yaxis
    else:
        raise ValueError(f'"which" must be "x" or "y", but got "{repr(which)}"')

    boxen = [
        t.get_tightbbox(fig.canvas.get_renderer()) for t in the_ax.get_ticklabels()
    ]

    if which == 'x':
        boxen = np.array([(b.x0, b.x1) for b in boxen])
    else:
        boxen = np.array([(b.y0, b.y1) for b in boxen])

    spacing = boxen[1:, 0] - boxen[:-1, 1]

    return np.min(spacing) < spacing_threshold


def rotate_ticklabels_on_collision(ax, which: str, angles: list, spacing_threshold=3):
    # lazy import of submodules seems to cause problems for matplotlib
    from matplotlib import pyplot as plt

    def set_rotation(the_ax, angle):
        for label in the_ax.get_ticklabels():
            label.set_rotation(angle)
            if which == 'y' and angle == 90:
                label.set_verticalalignment('center')
            elif which == 'x' and angle == 90:
                label.set_horizontalalignment('right')

    if which == 'x':
        the_ax = ax.xaxis
    elif which == 'y':
        the_ax = ax.yaxis
    else:
        raise ValueError(
            f'"which" argument must be "x" or "y", but got "{repr(which)}"'
        )

    set_rotation(the_ax, angles[0])
    if len(angles) == 1:
        return angles[0]

    a = angles[0]
    for angle in angles[1:]:
        plt.draw()

        if _has_tick_label_collision(ax, which, spacing_threshold):
            a = angle
            set_rotation(the_ax, angle)
        else:
            break
    return a


def xaxis_concise_dates(fig, ax, adjacent_offset: bool = True):
    """fuss with the dates on an x-axis."""

    # lazy import of submodules seems to cause problems for matplotlib
    from matplotlib import pyplot as plt

    formatter = mpl.dates.ConciseDateFormatter(
        mpl.dates.AutoDateLocator(), show_offset=True
    )

    if adjacent_offset:
        plt.xticks(rotation=0, ha='right')
    ax.xaxis.set_major_formatter(formatter)

    plt.draw()

    if adjacent_offset:
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[0] = f'{formatter.get_offset()} {labels[0]}'
        ax.set_xticklabels(labels)

        dx = 5 / 72.0
        dy = 0 / 72.0
        offset = mpl.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        for label in ax.get_xticklabels():
            label.set_transform(label.get_transform() + offset)

    return ax


def pcolormesh_df(
    df,
    vmin=None,
    vmax=None,
    rasterized=True,
    cmap=None,
    ax=None,
    xlabel=None,
    ylabel=None,
    title=None,
    norm=None,
    x_unit=None,
    x_places=None,
    y_unit=None,
    y_places=None,
):
    # lazy import of submodules seems to cause problems for matplotlib
    from matplotlib import pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots()

    X = df.columns.values
    Y = df.index.values

    drawing = ax.pcolormesh(
        X,
        Y,
        df.values,
        vmin=vmin,
        vmax=vmax,
        rasterized=rasterized,
        cmap=cmap,
        norm=norm,
        edgecolors='none',
        edgecolor='none',
    )

    if xlabel is not False:
        ax.set_xlabel(df.columns.name if xlabel is None else xlabel)

    if ylabel is not False:
        ax.set_ylabel(df.index.name if ylabel is None else ylabel)

    if title is not None:
        ax.set_title(title)

    if x_unit is not None:
        ax.xaxis.set_major_formatter(
            mpl.ticker.EngFormatter(unit=x_unit, useMathText=True, places=x_places)
        )
        rotate_ticklabels_on_collision(ax, 'x', [0, 25])

    if y_unit is not None:
        ax.yaxis.set_major_formatter(
            mpl.ticker.EngFormatter(unit=y_unit, useMathText=True, places=y_places)
        )
        rotate_ticklabels_on_collision(ax, 'y', [90, 65, 0])

    return drawing


def plot_spectrogram_heatmap_from_iq(
    iq: np.array,
    window: np.array,
    Ts: float,
    ax=None,
    vmin: float = None,
    cmap=None,
    time_span=(None, None),
) -> tuple[type_stubs.AxesType, type_stubs.DataFrameType]:
    # lazy import of submodules seems to cause problems for matplotlib
    from matplotlib import pyplot as plt
    
    index_span = (
        None if time_span[0] is None else int(np.rint(time_span[0] / Ts)),
        None if time_span[1] is None else int(np.rint(time_span[1] / Ts)),
    )

    iq = iq[index_span[0] : index_span[1]]

    spg = iq_to_stft_spectrogram(iq=iq, window=window, Ts=Ts, overlap=True)

    if cmap is None:
        cmap = mpl.cm.get_cmap('magma')

    c = pcolormesh_df(
        powtodB(spg.T),
        xlabel='Time elapsed (s)',
        ylabel='Baseband Frequency',
        y_unit='Hz',
        # x_unit='s',
        ax=ax,
        cmap=cmap,
        vmin=vmin,
    )

    freq_res = 1 / Ts / window.size

    if freq_res < 1e3:
        freq_res_name = f'{freq_res:0.1f}'
    elif freq_res < 1e6:
        freq_res_name = f'{freq_res/1e3:0.1f} kHz'
    elif freq_res < 1e9:
        freq_res_name = f'{freq_res/1e6:0.1f} MHz'
    else:
        freq_res_name = f'{freq_res/1e9:0.1f} GHz'

    plt.colorbar(
        c,
        cmap=cmap,
        ax=ax,
        label=f'Bin power (dBm/{freq_res_name})',
        # rasterized=True
    )

    return ax, spg


def plot_spectrogram_heatmap(
    spg: type_stubs.DataFrameType,
    Ts: float,
    ax=None,
    vmin: float = None,
    vmax: float = None,
    cmap=None,
    time_span=(None, None),
    transpose=False,
    colorbar=True,
    rasterized=True,
) -> tuple[type_stubs.AxesType, type_stubs.DataFrameType]:
    # lazy import of submodules seems to cause problems for matplotlib
    from matplotlib import pyplot as plt
    
    if cmap is None:
        cmap = mpl.cm.get_cmap('magma')

    if transpose:
        c = pcolormesh_df(
            powtodB(spg),
            ylabel='Time elapsed (s)',
            xlabel='Baseband Frequency',
            x_unit='Hz',
            # x_unit='s',
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=rasterized,
        )
    else:
        c = pcolormesh_df(
            powtodB(spg.T),
            xlabel='Time elapsed (s)',
            ylabel='Baseband Frequency',
            y_unit='Hz',
            # x_unit='s',
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=rasterized,
        )

    freq_res = 1 / Ts / spg.shape[1]

    if freq_res < 1e3:
        freq_res_name = f'{freq_res:0.1f}'
    elif freq_res < 1e6:
        freq_res_name = f'{freq_res/1e3:0.1f} kHz'
    elif freq_res < 1e9:
        freq_res_name = f'{freq_res/1e6:0.1f} MHz'
    else:
        freq_res_name = f'{freq_res/1e9:0.1f} GHz'

    if colorbar:
        plt.colorbar(
            c,
            cmap=cmap,
            ax=ax,
            label=f'Bin power (dBm/{freq_res_name})',
            # rasterized=True
        )

    return ax, spg


def plot_power_histogram_heatmap(
    rolling_histogram: type_stubs.DataFrameType,
    contiguous_threshold=None,
    log_counts=True,
    title: str = None,
    ylabel: str = None,
    xlabel: str = None,
    clabel: str = 'Count',
    xlim: tuple = None,
    ax=None,
    cbar=True,
    rasterized=True,
    x_unit=None,
    x_places=None,
):
    """plot a heat map of power histograms along the time axis, with color map intensity set by the counts.

    Args:
        rolling_histogram: histogram data, given along axis 0

        contiguous_threshold: plot gaps ()
    """
    # lazy import of submodules seems to cause problems for matplotlib
    from matplotlib import pyplot as plt

    if xlim is not None:
        rolling_histogram = rolling_histogram.loc[:, float(xlim[0]) : float(xlim[1])]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        try:
            fig = ax.get_figure()
        except BaseException:
            raise ValueError(str(locals()))

    if rolling_histogram.shape[0] == 0:
        raise EOFError

    index_type = type(rolling_histogram.index[0])

    # elif issubclass(index_type, pd.Timedelta):
    #     pass
    # else:
    #     raise ValueError(
    #         f"don't know how to handle index type {index_type} for 2D histogram over time"
    #     )

    # quantize the color map levels to the number of bins
    bad_color = '0.95'
    cmap = mpl.cm.get_cmap('magma')
    if rolling_histogram.shape[1] < cmap.N:
        subset = np.linspace(
            0, len(cmap.colors) - 1, rolling_histogram.shape[1], dtype=int
        )
        newcolors = np.array(cmap.colors)[subset].tolist()
        cmap = mpl.colors.ListedColormap(newcolors)
        cmap.set_bad(bad_color)

    if log_counts:
        if rolling_histogram.values.dtype == np.dtype('int64'):
            plot_norm = mpl.colors.LogNorm(vmin=1, vmax=rolling_histogram.max().max())
        else:
            plot_norm = mpl.colors.LogNorm(
                vmin=rolling_histogram[rolling_histogram > 0].min().min(),
                vmax=rolling_histogram.max().max(),
            )
    else:
        plot_norm = None

    pc_kws = dict(
        cmap=cmap,
        norm=plot_norm,
        rasterized=rasterized,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        ax=ax,
        x_unit=x_unit,
        x_places=x_places,
    )

    if issubclass(index_type, pd.Timestamp):
        # break into contiguous segments so that mpl will not project lines across
        # missing data

        if contiguous_threshold is not None:
            segments = contiguous_segments(
                rolling_histogram, 'Time', threshold=contiguous_threshold
            )
        else:
            segments = [rolling_histogram]

        for hist_sub in segments:
            c = pcolormesh_df(hist_sub.T, **pc_kws)

    elif issubclass(index_type, pd.Timedelta):
        if rolling_histogram.index[1] - rolling_histogram.index[0] < pd.Timedelta(3600):
            t = rolling_histogram.index.total_seconds() / 3600
        else:
            t = rolling_histogram.index.total_seconds()

        hist_sub = pd.DataFrame(
            rolling_histogram.values, index=t, columns=rolling_histogram.columns
        )

        c = pcolormesh_df(hist_sub.T, **pc_kws)
        # c = pcolormesh_df(rolling_histogram.T, **pc_kws)

    else:
        c = pcolormesh_df(rolling_histogram.T, **pc_kws)

    if cbar and not log_counts:
        cb = fig.colorbar(
            c,
            cmap=cmap,
            ax=ax,
            extend='min',
            extendrect=True,
            # extendfrac='auto',
            # cax = fig.add_axes([1.02, 0.152, 0.03, 0.7])
        )

        formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        cb.ax.yaxis.set_major_formatter(formatter)
        cb.ax.ticklabel_format(style='sci', scilimits=(6, 6))
        cb.ax.yaxis.get_offset_text().set_position((0, 1.01))
        cb.ax.yaxis.get_offset_text().set_horizontalalignment('left')
        cb.ax.yaxis.get_offset_text().set_verticalalignment('bottom')

        cb.set_label(
            clabel,
            labelpad=-16,
            y=-0.08,
            # x=-1,
            rotation=0,
            va='top',
            ha='right',
        )

    elif cbar:
        cbar_cmap = mpl.colors.ListedColormap(cmap.colors.copy())
        cbar_cmap.set_under(bad_color)
        cbar_cmap.set_bad(bad_color)

        cb = fig.colorbar(
            c,
            cmap=cbar_cmap,
            ax=ax,
            extend='min',
            extendrect=True,
            extendfrac=0.05,
            # cax = fig.add_axes([1.02, 0.152, 0.03, 0.75])
        )

        # add in the extension
        extension_length = cb._get_extension_lengths(cb.extendfrac, True, True)[1]
        cb._boundaries = np.array(
            [np.nan]
            + list(
                np.linspace(
                    cb._boundaries[0], cb._boundaries[1], cb._boundaries.size - 1
                )
            )
        )
        cb._values = np.array(
            [np.nan]
            + list(np.linspace(cb._values[0], cb._values[1], cb._values.size - 1))
        )

        cb._do_extends(cb._get_extension_lengths(extension_length, True, True))

        cb.ax.text(
            1,
            -extension_length / 2,
            '- 0',
            ha='left',
            va='center',
            transform=cb.ax.transAxes,
        )

        formatter = mpl.ticker.LogFormatterSciNotation(
            minor_thresholds=(1, 2, 5), labelOnlyBase=False
        )

        cb.ax.yaxis.set_major_formatter(formatter)
        cb.ax.yaxis.set_minor_formatter(formatter)

        # cb.ax.xaxis.set_major_locator(mpl.ticker.AutoLocator())
        # cb.ax.xaxis.set_minor_formatter(mpl.ticker.StrMethodFormatter(f''))

        cb.set_label(
            clabel,
            labelpad=-16,
            y=-0.08,
            # x=-1,
            rotation=0,
            va='top',
            ha='right',
        )
    else:
        cb = None

    # X axis formatting
    if issubclass(index_type, (pd.Timestamp)):
        xaxis_concise_dates(plt.gcf(), ax)
    else:
        plt.draw()
        # labels = [f"{l.get_text()}:00" for l in ax.get_xticklabels()]
        # ax.set_xticklabels(labels)
    # @mpl.ticker.FuncFormatter
    # def minor_formatter(x, pos):
    #     exp = int(np.trunc(np.log10(x)))
    #     return rf'${x/10**(exp-1):0.0f}$'

    if cb is not None and cb.vmax / cb.vmin < 1e3:
        cb.ax.yaxis.set_minor_formatter(formatter)
        pass
        # for label in cb.ax.yaxis.get_minorticklabels()[1::2]:
        #     label.set_visible(False)

    return ax, c


def plot_power_ccdf(
    iq,
    Ts,
    Tavg=None,
    random_offsets=False,
    bins=None,
    scale='gamma-qq',
    major_ticks=12,
    ax=None,
    label=None,
):
    # lazy import of submodules seems to cause problems for matplotlib
    from matplotlib import pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots()

    if Tavg is None:
        Navg = 1
        power_dB = envtodB(iq)
    else:
        Navg = int(Tavg / Ts)
        power_dB = powtodB(
            iq_to_bin_power(
                iq, Ts=Ts, Tbin=Tavg, randomize=random_offsets, truncate=True
            )
        )

    if bins is None:
        bins = np.arange(power_dB.min(), power_dB.max() + 0.01, 0.01)
    if np.isscalar(bins):
        bins = np.linspace(power_dB.min(), power_dB.max(), bins)
    else:
        bins = np.array(bins)

    ccdf = sample_ccdf(power_dB, bins)
    ax.plot(ccdf, bins, label=label)  # Path(DATA_FILE).parent.name)

    if scale == 'gamma-qq':
        ax.set_xscale(scale, k=Navg, major_ticks=major_ticks, db_ordinal=True)
    else:
        ax.set_xscale(scale)
    ax.legend()

    return ax, ccdf, bins
