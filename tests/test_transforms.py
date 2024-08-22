"""eventual test cases"""

from iqwaveform import dBtopow, powtodB, envtopow
import pandas as pd
import numpy as np

def test_transform_int():
    assert powtodB(1) == 0

def test_transform_float():
    assert powtodB(1.0) == 0

def test_transform_series():
    s = pd.Series([1, 10, 100])
    expect = pd.Series([0, 10, 20])
    ret = powtodB(s)
    return np.allclose(expect.values, ret.values)
