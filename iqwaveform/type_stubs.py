from __future__ import annotations
import typing
from typing_extensions import TypeAlias

if typing.TYPE_CHECKING:
    # bury this type checking in here to avoid lengthening the import time of iqwaveform
    # if cupy isn't installed
    try:
        import cupy as cp
    except ModuleNotFoundError:
        import numpy as cp
    import numpy as np

    import pandas as pd
    from matplotlib import axes
    import matplotlib as mpl

# union of supported array types
Array: TypeAlias = typing.Union['cp.ndarray','np.ndarray']
DataFrameType: TypeAlias = 'pd.DataFrame'
SeriesType: TypeAlias = 'pd.Series'
IndexType: TypeAlias = 'pd.Index'
AxisType: TypeAlias = 'axes.Axes'
LocatorType: TypeAlias = 'mpl.ticker.MaxNLocator'
ArrayOrPandas: TypeAlias = typing.Union[Array, SeriesType, DataFrameType]
