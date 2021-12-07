import numpy as np
from mindspore import ms_function, ops, Tensor, dtype


@ms_function
def expand_tensor(a, b):
    out = ops.tile(a, b)
    return out


def test_tile_eliminate():
    """
    Feature: tile_eliminate
    Description: All value of multiplier is '1' but length of multiplier is greater than tensor dims, can't do eliminate
    Expectation: success
    """
    tensor_ = Tensor(np.ndarray([1, 448, 448]), dtype=dtype.float32)
    out = ops.tile(tensor_, (1, 1, 1))
    assert out.shape == (1, 448, 448)
    out = ops.tile(tensor_, (1, 1, 1, 1))
    assert out.shape == (1, 1, 448, 448)
    out = expand_tensor(tensor_, (1, 1, 1))
    assert out.shape == (1, 448, 448)
    out = expand_tensor(tensor_, (1, 1, 1, 1))
    assert out.shape == (1, 1, 448, 448)
