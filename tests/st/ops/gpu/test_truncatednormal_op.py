import pytest
import numpy as np
import mindspore
from mindspore import context
from mindspore import Tensor
from mindspore.ops.operations.random_ops import TruncatedNormal
from mindspore import nn


class RandomTruncatedNormal(nn.Cell):
    def __init__(self):
        super(RandomTruncatedNormal, self).__init__()
        self.truncatednormal = TruncatedNormal()

    def construct(self, shape):
        return self.truncatednormal(shape)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_truncatednormal_graph():
    """
    Feature: truncatednormal gpu kernel
    Description: Follow normal distribution, with in 2 standard deviations.
    Expectation: match to tensorflow benchmark.
    """

    shape = Tensor([2, 2], dtype=mindspore.int32)
    truncatednormal_test = RandomTruncatedNormal()
    expect = np.array([2, 2])

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    output = truncatednormal_test(shape)
    assert (output.shape == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_truncatednormal_pynative():
    """
    Feature: truncatednormal gpu kernel
    Description: Follow normal distribution, with in 2 standard deviations.
    Expectation: match to tensorflow benchmark.
    """

    shape = Tensor([2, 2], dtype=mindspore.int32)
    truncatednormal_test = RandomTruncatedNormal()
    expect = np.array([2, 2])

    context.set_context(mode=context.PYNATIVE_MODE)
    output = truncatednormal_test(shape)
    assert (output.shape == expect).all()
