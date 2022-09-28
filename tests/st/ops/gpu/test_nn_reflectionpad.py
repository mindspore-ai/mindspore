import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
import pytest


class Net1d(nn.Cell):
    def __init__(self, padding):
        super(Net1d, self).__init__()
        self.pad = nn.ReflectionPad1d(padding)

    def construct(self, x):
        return self.pad(x)


class Net2d(nn.Cell):
    def __init__(self, padding):
        super(Net2d, self).__init__()
        self.pad = nn.ReflectionPad2d(padding)

    def construct(self, x):
        return self.pad(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reflection_pad_1d():
    """
    Feature: ReflectionPad1d
    Description: Infer process of ReflectionPad1d with 2 types of parameters.
    Expectation: success
    """
    # Test functionality with 3D tensor input
    x = Tensor(np.array([[[0, 1, 2, 3], [4, 5, 6, 7]]]).astype(np.float32))
    padding = (3, 1)
    net = Net1d(padding)
    output = net(x)
    expected_output = Tensor(np.array([[[3, 2, 1, 0, 1, 2, 3, 2],
                                        [7, 6, 5, 4, 5, 6, 7, 6]]]).astype(np.float32))

    assert np.array_equal(output, expected_output)

    padding = 2
    expected_output = Tensor(np.array([[[2, 1, 0, 1, 2, 3, 2, 1],
                                        [6, 5, 4, 5, 6, 7, 6, 5]]]).astype(np.float32))
    net = Net1d(padding)
    output = net(x)
    assert np.array_equal(output, expected_output)

    # Test functionality with 2D tensor as input
    x = Tensor(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]).astype(np.float16))
    padding = (3, 1)
    net = Net1d(padding)
    output = net(x)
    expected_output = Tensor(np.array([[3, 2, 1, 0, 1, 2, 3, 2],
                                       [7, 6, 5, 4, 5, 6, 7, 6]]).astype(np.float16))
    assert np.array_equal(output, expected_output)

    padding = 2
    expected_output = Tensor(np.array([[2, 1, 0, 1, 2, 3, 2, 1],
                                       [6, 5, 4, 5, 6, 7, 6, 5]]).astype(np.float16))
    net = Net1d(padding)
    output = net(x)
    assert np.array_equal(output, expected_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reflection_pad_2d():
    r"""
    Feature: ReflectionPad2d
    Description: Infer process of ReflectionPad2d with three type parameters.
    Expectation: success
    """
    # Test functionality with 4D tensor as input
    x = Tensor(np.array([[[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]]).astype(np.int32))
    padding = (1, 1, 2, 0)
    net = Net2d(padding)
    output = net(x)
    expected_output = Tensor(np.array([[[[7, 6, 7, 8, 7], [4, 3, 4, 5, 4], [1, 0, 1, 2, 1],
                                         [4, 3, 4, 5, 4], [7, 6, 7, 8, 7]]]]).astype(np.int32))
    assert np.array_equal(output, expected_output)

    padding = 2
    net = Net2d(padding)
    output = net(x)
    expected_output = Tensor(np.array([[[[8, 7, 6, 7, 8, 7, 6], [5, 4, 3, 4, 5, 4, 3],
                                         [2, 1, 0, 1, 2, 1, 0], [5, 4, 3, 4, 5, 4, 3],
                                         [8, 7, 6, 7, 8, 7, 6], [5, 4, 3, 4, 5, 4, 3],
                                         [2, 1, 0, 1, 2, 1, 0]]]]).astype(np.int32))
    assert np.array_equal(output, expected_output)

    # Test functionality with 3D tensor as input
    x = Tensor(np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]).astype(np.float32))
    padding = (1, 1, 2, 0)
    net = Net2d(padding)
    output = net(x)
    expected_output = Tensor(np.array([[[7, 6, 7, 8, 7], [4, 3, 4, 5, 4], [1, 0, 1, 2, 1],
                                        [4, 3, 4, 5, 4], [7, 6, 7, 8, 7]]]).astype(np.float32))
    assert np.array_equal(output, expected_output)

    padding = 2
    net = Net2d(padding)
    output = net(x)
    expected_output = Tensor(np.array([[[8, 7, 6, 7, 8, 7, 6], [5, 4, 3, 4, 5, 4, 3],
                                        [2, 1, 0, 1, 2, 1, 0], [5, 4, 3, 4, 5, 4, 3],
                                        [8, 7, 6, 7, 8, 7, 6], [5, 4, 3, 4, 5, 4, 3],
                                        [2, 1, 0, 1, 2, 1, 0]]]).astype(np.float32))
    assert np.array_equal(output, expected_output)
