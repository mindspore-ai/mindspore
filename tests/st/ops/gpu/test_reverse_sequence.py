import pytest
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, seq_dim, batch_dim):
        super(Net, self).__init__()
        self.reverse_sequence = P.ReverseSequence(
            seq_dim=seq_dim, batch_dim=batch_dim)

    @ms_function
    def construct(self, x, seq_lengths):
        return self.reverse_sequence(x, seq_lengths)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_int8():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.int8)
    seq_lengths = np.array([1, 2, 3]).astype(np.int32)
    seq_dim = 0
    batch_dim = 1
    net = Net(seq_dim, batch_dim)
    output = net(Tensor(x), Tensor(seq_lengths))
    expected = np.array([[1, 5, 9], [4, 2, 6], [7, 8, 3]]).astype(np.int8)
    assert np.array_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_int32():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.int32)
    seq_lengths = np.array([1, 2, 3]).astype(np.int64)
    seq_dim = 1
    batch_dim = 0
    net = Net(seq_dim, batch_dim)
    output = net(Tensor(x), Tensor(seq_lengths))
    expected = np.array([[1, 2, 3], [5, 4, 6], [9, 8, 7]]).astype(np.int32)
    assert np.array_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_float32():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([[[[1, 2], [3, 4]],
                   [[5, 6], [7, 8]],
                   [[9, 10], [11, 12]],
                   [[13, 14], [15, 16]]],
                  [[[17, 18], [19, 20]],
                   [[21, 22], [23, 24]],
                   [[25, 26], [27, 28]],
                   [[29, 30], [31, 21]]]]).astype(np.float32)
    seq_lengths = np.array([2, 2, 2, 2]).astype(np.int64)
    seq_dim = 0
    batch_dim = 1
    net = Net(seq_dim, batch_dim)
    output = net(Tensor(x), Tensor(seq_lengths))
    expected = np.array([[[[17., 18.], [19., 20.]],
                          [[21., 22.], [23., 24.]],
                          [[25., 26.], [27., 28.]],
                          [[29., 30.], [31., 21.]]],
                         [[[1., 2.], [3., 4.]],
                          [[5., 6.], [7., 8.]],
                          [[9., 10.], [11., 12.]],
                          [[13., 14.], [15., 16.]]]]).astype(np.float32)
    assert np.array_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_float64_0_dim():
    """
    Test added to test for 0 seq len edge case
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([[[[1, 2], [3, 4]],
                   [[5, 6], [7, 8]],
                   [[9, 10], [11, 12]],
                   [[13, 14], [15, 16]]],
                  [[[17, 18], [19, 20]],
                   [[21, 22], [23, 24]],
                   [[25, 26], [27, 28]],
                   [[29, 30], [31, 21]]]]).astype(np.float32)
    seq_lengths = np.array([2, 2, 0, 0]).astype(np.int64)
    seq_dim = 2
    batch_dim = 1
    net = Net(seq_dim, batch_dim)
    output = net(Tensor(x), Tensor(seq_lengths))
    expected = np.array([[[[3., 4.], [1., 2.]],
                          [[7.,  8.], [5., 6.]],
                          [[9., 10.], [11., 12.]],
                          [[13., 14.], [15., 16.]]],
                         [[[19., 20.], [17., 18.]],
                          [[23., 24.], [21., 22.]],
                          [[25., 26.], [27., 28.]],
                          [[29., 30.], [31., 21.]]]]).astype(np.float32)
    assert np.array_equal(output.asnumpy(), expected)
