import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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
        l.get_tightbbox(fig.canvas.get_renderer())
        for l in the_ax.get_ticklabels()
    ]
    
    if which == 'x':
        boxen = np.array([(b.x0, b.x1) for b in boxen])
    else:
        boxen = np.array([(b.y0, b.y1) for b in boxen])    

    spacing = boxen[1:,0] - boxen[:-1,1]


    return np.min(spacing) < spacing_threshold

def rotate_ticklabels_on_collision(ax, which: str, angles: list, spacing_threshold=3):
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
        raise ValueError(f'"which" argument must be "x" or "y", but got "{repr(which)}"')

    set_rotation(the_ax, angles[0])
    if len(angles) == 1:
        return angles[0]

    a = angles[0]
    for angle in angles[1:]:
        plt.draw()

        if _has_tick_label_collision (ax, which, spacing_threshold):
            a = angle
            set_rotation(the_ax, angle)
        else:
            break
    return a


def xaxis_concise_dates(fig, ax, adjacent_offset: bool = True):
    """fuss with the dates on an x-axis."""
    formatter = mpl.dates.ConciseDateFormatter(
        mpl.dates.AutoDateLocator(), show_offset=True
    )

    if adjacent_offset:
        plt.xticks(rotation=0, ha="right")
    ax.xaxis.set_major_formatter(formatter)

    plt.draw()

    if adjacent_offset:
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[0] = f"{formatter.get_offset()} {labels[0]}"
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
    y_places=None
):

    if ax is None:
        fig, ax = plt.subplots()
    
    drawing = ax.pcolormesh(
        df.columns,
        df.index,        
        df,
        vmin=vmin,
        rasterized=rasterized,
        cmap=cmap,
        norm=norm
    )

    if xlabel is not False:
        ax.set_xlabel(df.columns.name if xlabel is None else xlabel)

    if ylabel is not False:
        ax.set_ylabel(df.index.name if ylabel is None else ylabel)

    if title is not None:
        ax.set_title(title)

    if x_unit is not None:
        ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter(unit=x_unit, useMathText=True, places=x_places))
        rotate_ticklabels_on_collision(ax, 'x', [0, 25])

    if y_unit is not None:
        ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter(unit=y_unit, useMathText=True, places=y_places))        
        rotate_ticklabels_on_collision(ax, 'y', [90, 65, 0])

    return drawing




def plot_power_histogram_heatmap(
    rolling_histogram: pd.DataFrame,
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
    x_places=None
):

    """plot a heat map of power histograms along the time axis, with color map intensity set by the counts.

    Args:
        rolling_histogram: histogram data, given along axis 0

        contiguous_threshold: plot gaps ()
    """

    if xlim is not None:
        rolling_histogram = rolling_histogram.loc[:, float(xlim[0]) : float(xlim[1])]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        try:
            fig = ax.get_figure()
        except:
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
    cmap = mpl.cm.get_cmap("magma")
    if rolling_histogram.shape[1] < cmap.N:
        subset = np.linspace(0, len(cmap.colors) - 1, rolling_histogram.shape[1], dtype=int)
        newcolors = np.array(cmap.colors)[subset].tolist()
        cmap = mpl.colors.ListedColormap(newcolors)
        cmap.set_bad(bad_color)

    if log_counts:
        if rolling_histogram.values.dtype == np.dtype("int64"):
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
        x_places=x_places
    )

    if issubclass(index_type, pd.Timestamp):
        # break into contiguous segments so that mpl will not project lines across
        # missing data

        if contiguous_threshold is not None:
            segments = contiguous_segments(
                rolling_histogram, "Time", threshold=contiguous_threshold
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
            rolling_histogram,
            index=t
        )

        c = pcolormesh_df(hist_sub.T, **pc_kws)

    else:
        c = pcolormesh_df(rolling_histogram.T, **pc_kws)

        # print(rolling_histogram.index[0])
        # raise ValueError(f"unrecognized Time index type {index_type}")

    if cbar and not log_counts:
        cb = fig.colorbar(
            c,
            cmap=cmap,
            ax=ax,
            extend='min',
            extendrect=True
            # extendfrac='auto',            
            # cax = fig.add_axes([1.02, 0.152, 0.03, 0.7])
        )

        formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        cb.ax.yaxis.set_major_formatter(formatter)
        cb.ax.ticklabel_format(style="sci", scilimits=(6, 6))
        cb.ax.yaxis.get_offset_text().set_position((0, 1.01))
        cb.ax.yaxis.get_offset_text().set_horizontalalignment("left")
        cb.ax.yaxis.get_offset_text().set_verticalalignment("bottom")

        cb.set_label(
            clabel,
            labelpad=-16,
            y=-0.08,
            # x=-1,
            rotation=0,
            va="top",
            ha="right",
        )

    elif cbar:
        cbar_cmap = cmap.copy()
        cbar_cmap.set_under(bad_color)
        cbar_cmap.set_bad(bad_color)

        cb = fig.colorbar(
            c,
            cmap=cbar_cmap,
            ax=ax,
            extend='min',
            extendrect=True,
            extendfrac=.05,
            # cax = fig.add_axes([1.02, 0.152, 0.03, 0.75])
        )

        # add in the extension
        extension_length = cb._get_extension_lengths(cb.extendfrac, True, True)[1]
        cb._boundaries = np.array([np.nan]+list(np.linspace(cb._boundaries[0], cb._boundaries[1], cb._boundaries.size-1)))
        cb._values = np.array([np.nan]+list(np.linspace(cb._values[0], cb._values[1], cb._values.size-1)))
        cb._do_extends(cb._get_extension_lengths(extension_length, True, True))

        cb.ax.text(1, -extension_length/2, '- 0', ha='left', va='center', transform=cb.ax.transAxes)

        formatter = mpl.ticker.LogFormatterSciNotation(minor_thresholds = (1, 2, 5), labelOnlyBase=False)

        cb.ax.yaxis.set_major_formatter(formatter)
        cb.ax.yaxis.set_minor_formatter(formatter)

        # cb.ax.xaxis.set_major_locator(mpl.ticker.AutoLocator())
        # cb.ax.xaxis.set_minor_formatter(mpl.ticker.StrMethodFormatter(f'')) 
        
        cb.set_label(
            clabel,
            labelpad=-16,
            y=-.08,
            # x=-1,
            rotation=0,
            va="top",
            ha="right",
        )

    # X axis formatting
    if issubclass(index_type, (pd.Timestamp, pd.Timedelta)):
        xaxis_concise_dates(plt.gcf(), ax)
    else:
        plt.draw()
        # labels = [f"{l.get_text()}:00" for l in ax.get_xticklabels()]
        # ax.set_xticklabels(labels)
    # @mpl.ticker.FuncFormatter
    # def minor_formatter(x, pos):
    #     exp = int(np.trunc(np.log10(x)))
    #     return rf'${x/10**(exp-1):0.0f}$'


    if cb.vmax / cb.vmin < 1e3:
        cb.ax.yaxis.set_minor_formatter(formatter)
        pass
        # for label in cb.ax.yaxis.get_minorticklabels()[1::2]:
        #     label.set_visible(False)

    return ax, cb
