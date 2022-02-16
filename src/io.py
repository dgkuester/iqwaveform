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
):
    x_split, center_freqs, Ts = read_iq(**locals())

    df = pd.DataFrame(dict(zip(center_freqs / 1e9, x_split)))
    df.columns.name = "Frequency (GHz)"
    df.index = df.index * Ts
    df.index.name = "Time elapsed (s)"

    return df


def resample_iq(iq: np.array, Ts, scale, axis=0):
    N = int(np.round(iq.shape[0] * scale))
    return signal.resample(iq, num=N, axis=axis), Ts / scale
