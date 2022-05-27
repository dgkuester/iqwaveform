# routines for reading spectrum monitoring data files

import numpy as np
import pandas as pd
import json
from scipy import signal
from pathlib import Path


def read_sigmf_iq_metadata(metadata_fn) -> tuple((pd.DataFrame, float)):
    with open(metadata_fn, "r") as fd:
        metadata = json.load(fd)

    df = pd.DataFrame(metadata["captures"])

    df.columns = [n.replace("core:", "") for n in df.columns]

    return (
        dict(df.set_index("sample_start").frequency),
        dict(df.set_index("sample_start").datetime),
        metadata["global"]["core:sample_rate"],
    )


def read_iq(
    metadata_path: str,
    force_sample_rate: float = None,
    sigmf_data_ext=".npy",
    stack=False,
):
    metadata_path = Path(metadata_path)

    """pack a DataFrame with data read from a SigMF modified for npy file format"""
    center_freqs, timestamps, sample_rate = read_sigmf_iq_metadata(metadata_path)

    if force_sample_rate is not None:
        sample_rate = force_sample_rate

    if sigmf_data_ext == ".npy":
        data_fn = metadata_path.with_suffix(".sigmf-data.npy")

        x = np.load(data_fn)
    else:
        raise TypeError(f"SIGMF data extension {sigmf_data_ext} not supported")

    x_split = np.array_split(x, list(center_freqs.keys())[1:])

    if stack:
        x_split = np.vstack(x_split).T

    return (x_split, np.array(list(center_freqs.values())), 1.0 / sample_rate)


def read_iq_to_df(
    metadata_path: str, force_sample_rate: float = None, sigmf_data_ext=".npy"
) -> np.array:
    x_split, center_freqs, Ts = read_iq(**locals())

    return iq_to_frame(
        x_split,
        Ts,
        columns=pd.Index(center_freqs/1e9), name='Frequency (Hz)'
    )


def iq_to_frame(iq: np.array, Ts: float, columns:pd.Index=None) -> tuple((pd.Series, pd.DataFrame)):
    """packs IQ data into a pandas Series or DataFrame object.

    The input waveform `iq` may have shape (N,) or (N,M), representing a single
    waveform or M different waveforms, respectively.

    Args:
        iq: Complex-valued time series representing an IQ waveform.
        Ts: The sample period of the IQ waveform.
        columns: The list of column names to use if `iq` has 2-dimensions

    Returns:
        If iq.ndim == 1, then pd.Series, otherwise pd.DataFrame, with a time index
    """

    if iq.ndim == 2:
        if columns is None:
            columns = np.arange(iq.shape[1])
        obj = pd.DataFrame(dict(zip(columns, iq)))
    elif iq.ndim == 1:
        obj = pd.Series(iq)
    else:
        raise TypeError(f'iq must have 1 or 2 dimensions')

    obj.index = pd.Index(
        np.linspace(0, Ts*iq.shape[0], Ts, endpoint=False),
        name = "Time elapsed (s)"
    )        

    return obj


def resample_iq(iq: np.array, Ts, scale, axis=0):
    N = int(np.round(iq.shape[0] * scale))
    return signal.resample(iq, num=N, axis=axis), Ts / scale
