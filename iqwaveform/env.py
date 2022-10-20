# A signal processing environment for notebooks
import sys

sys.path = [".."] + sys.path

import warnings

warnings.simplefilter("ignore")

# Set up the plotting environment for notebooks that convert cleanly
# to pdf or html output.
import numpy as np
from scipy import signal
from matplotlib import pylab, mlab

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import rcParams, rc

import IPython
import IPython.display
from IPython.display import display, HTML, set_matplotlib_formats
from IPython.core.pylabtools import figsize, getfigs

from importlib import reload
import ipywidgets as widgets
import pandas as pd
from pathlib import Path
import seaborn as sns
import datetime

import functools

_captions = {}

from matplotlib.backends import backend_svg
import functools

@functools.wraps(backend_svg.FigureCanvasSVG.print_svg)
def print_svg(self, *a, **k):
    def guess_title(fig):
        if self.figure._suptitle is not None:
            return self.figure._suptitle.get_text()

        for ax in self.figure.get_axes()[::-1]:
            title_ = ax.get_title()
            if title_:
                return title_
        else:
            return "untitled"

    def title_to_label(title_):
        """replace 1 or more non-alphanumeric characters with '-'"""
        import re, string

        pattern = re.compile(r"[\W_]+")
        return pattern.sub("-", title_).lower()

    k = dict(k)
    label = title_to_label(guess_title(self.figure))
    caption_text = _captions.get(id(self.figure), "")
    title_ = f"{label}##{caption_text}" if caption_text else label
    k.setdefault("metadata", {})["Title"] = title_

    return backend_svg.FigureCanvasSVG._print_svg(self, *a, **k)

backend_svg.FigureCanvasSVG.print_svg, backend_svg.FigureCanvasSVG._print_svg = print_svg, backend_svg.FigureCanvasSVG.print_svg

@functools.wraps(IPython.display.set_matplotlib_formats)
def set_matplotlib_formats(formats, *args, **kws):
    """apply wrappers to inject title (from figure or axis titles) and caption (from set_caption metadata),
    when available, into image 'Title' metadata
    """

    IPython.display.set_matplotlib_formats(formats, *args, **kws)

    # monkeypatch IPython's internal print_figure to include title metadata
    from IPython.core import pylabtools

    pylabtools = reload(pylabtools)

    def guess_title(fig):
        if fig._suptitle is not None:
            return fig._suptitle.get_text()

        for ax in fig.get_axes()[::-1]:
            title_ = ax.get_title()
            if title_:
                return title_
        else:
            return "untitled"

    def title_to_label(title_):
        """replace 1 or more non-alphanumeric characters with '-'"""
        import re, string

        pattern = re.compile(r"[\W_]+")
        return pattern.sub("-", title_).lower()

    @functools.wraps(pylabtools.print_figure)
    def wrapper(fig, fmt='png', *a, **k):
        k = dict(k)
        label = title_to_label(guess_title(fig))
        caption_text = _captions.get(id(fig), "")

        ret = pylabtools._print_figure(fig, fmt=fmt, *a, **k)

        markup = f'<tt>{label}.{fmt}:</tt>{"<br>"+caption_text if caption_text else " (no caption data)"}'
        display(HTML(markup))

        return ret

    pylabtools.print_figure, pylabtools._print_figure = wrapper, pylabtools.print_figure

# automatically rasterize complicated artists
# @functools.wraps(mpl.axes.Axes.__init__)
# def __axes_init__ (*args, **kws):
#     kws.setdefault('rasterization_zorder', 2.005)
#     return mpl.axes.Axes.__init__(*args, **kws)
# mpl.axes.Axes.__init__ = __axes_init__

# requires pandas >= 1.0.0
convert_datetime = mpl.units.registry[np.datetime64]

def set_font_size(size):
    FONT_KEYS = [
        'axes.labelsize',
        'axes.titlesize',
        'figure.titlesize',
        'font.size',
        'legend.fontsize',
        'legend.title_fontsize',
        'xtick.labelsize',
        'xtick.major.size',
        'xtick.minor.size',
        'ytick.labelsize',
        'ytick.major.size',
        'ytick.minor.size'
    ]


    rcParams.update(rcParams.fromkeys(FONT_KEYS, size))

def set_caption(*args):
    """sets the caption in a jupyter notebook for the

    Usage: either set_caption(fig, text) or set_caption(text) to use the current figure
    """
    global _captions

    if len(args) == 1:
        fig, text = pylab.gcf(), args[0]
    elif len(args) == 2:
        fig, text = args
    else:
        raise ValueError(f"expected 1 or 2 args, but got {len(args)}")

    _captions[id(fig)] = text


time_format = "%Y-%m-%d %H:%M:%S"

# what IPython uses to display in the browser.
# IMPORTANT: pass rasterize=True in calls to plot, or the notebooks get very large!

# plot settings mostly designed for IEEE publication styles
sns.set(context="paper", style="ticks", font_scale=1)


def _set_colors_from_dict(color_dict, linestyle_cycle=None):
    from cycler import cycler

    if linestyle_cycle is not None:
        cyc = cycler(linestyle=linestyle_cycle) * cycler(color=color_dict)
    else:
        cyc = cycler(color=color_dict)

    rc("axes", prop_cycle=cyc)

    rc("patch", facecolor=list(color_dict.keys())[0])

    for color, code in color_dict.items():
        # map the colors into color codes like 'k', 'b', 'c', etc
        rgb = mpl.colors.colorConverter.to_rgb(color)
        mpl.colors.colorConverter.colors[code] = rgb
        mpl.colors.colorConverter.cache[code] = rgb


# colorblind-accessible palette from http://mkweb.bcgsc.ca/colorblind/palettes.mhtml#page-container
# _set_colors_from_dict(
#     {
#         '#2271B2': 'b',
#         '#3DB7E9': 'c',
#         '#359B73': 'g',
#         '#f0e442': 'y',
#         '#e69f00': 'o',
#         '#F748A5': 'm',
#         '#d55e00': 'r',
#         '#000000': 'k',
#     },
#     ['-', '--', '-.', ':']
# )
sns.set_palette("colorblind", 7, color_codes=True)
rc(
    "axes",
    prop_cycle=mpl.cycler(linestyle=["-", ":", "--"]) * rcParams["axes.prop_cycle"],
)


# concise date formatting by default
converter = mpl.dates.ConciseDateConverter()
mpl.units.registry[np.datetime64] = converter
mpl.units.registry[datetime.date] = converter
mpl.units.registry[datetime.datetime] = converter

rc(
    "font",
    family=["serif"],
    serif=["Times New Roman"],
    weight="normal",
    cursive="Freestyle Script",
)

# support for TeX-like math expressions
# (without the slowdown of usetex=True)
rc(
    "mathtext",
    fontset="custom",
    it="serif:italic",
    rm="serif:normal",
    bf="serif:bold",
    default="it",
)

rc("axes", labelweight="regular", **{"spines.top": False, "spines.right": False})

# tighten up saved figures for publication
rc(
    "savefig",
    bbox="standard",
    pad_inches=0,
    facecolor="none",  # equivalent to prior frameon=False
    #    transparent=False
)

# tighten up the legend
rc(
    "legend",
    handletextpad=0.2,
    labelspacing=0.005,
    borderaxespad=0.05,
    columnspacing=0.5,
    handlelength=1.25,
    edgecolor="k",
)

rc("lines", linewidth=0.5)

figsize_fullwidth = np.array([6.5, 2.35])
figsize_halfwidth = np.array([3.2, 2.35])

rc(
    "figure",
    figsize=figsize_fullwidth,  # autolayout=False,
    titlesize=10,
    dpi=300,
    **{
        "constrained_layout.use": True,
        "constrained_layout.h_pad": 0,
        "constrained_layout.w_pad": 0,
        #                 'subplot.left'    : 0.,  # the left side of the subplots of the figure
        #                 'subplot.right'   : 1.,    # the right side of the subplots of the figure
        #                 'subplot.bottom'  : 0.11,    # the bottom of the subplots of the figure
        #                 'subplot.top'     : 0.88,   # the top of the subplots of the figure
        #                 'subplot.wspace':  0,
        #                 'subplot.hspace': 0,
    },
)

rc("svg", fonttype="none")

font = mpl.font_manager.findfont(mpl.font_manager.FontProperties(family=["serif"]))

set_matplotlib_formats("svg")
set_font_size(10)
