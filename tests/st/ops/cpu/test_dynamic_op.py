# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from tests.mark_utils import arg_mark

import sys
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.composite import GradOperation

context.set_context(mode=context.GRAPH_MODE,
                    device_target="CPU")


class ConcatOffsetNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.unique = P.Unique()
        self.concat_offset = G.ConcatOffset(3, 0)
        self.reshape = P.Reshape()

    def construct(self, x, y, z):
        x = self.reshape(self.unique(x)[0], (-1, 1, 2, 1))
        y = self.reshape(self.unique(y)[0], (-1, 1, 2, 1))
        z = self.reshape(self.unique(z)[0], (-1, 1, 2, 1))
        out = self.concat_offset((x, y, z))
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_concat_offset_dynamic_cpu():
    """
    /// Feature: Concatoffset op dynamic shape
    /// Description: Concatoffset forward with dynamic shape
    /// Expectation: Euqal to expected value
    """
    if sys.platform != 'linux':
        return
    x = Tensor(np.array([1, 2, 3, 4, 5, 6]), mstype.float32)
    x2 = Tensor(np.array([1, 2, 3, 4, 5, 6]), mstype.float32)
    x3 = Tensor(np.array([1, 2, 3, 4, 5, 6]), mstype.float32)
    net = ConcatOffsetNet()
    out = net(x, x2, x3)
    expect = np.array([[0, 0, 0, 0],
                       [3, 0, 0, 0],
                       [6, 0, 0, 0]])
    if isinstance(out, tuple):
        assert (np.array(out) == expect).all()
    else:
        assert (out.asnumpy() == expect).all()


class SliceNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.slice = P.Slice()

    def construct(self, x, begin, size):
        return self.slice(x, begin, size)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_begin_size_tensor_cpu():
    """
    /// Feature: Slice op dynamic shape
    /// Description: Slice forward with dynamic shape
    /// Expectation: Euqal to expected value
    """
    if sys.platform != 'linux':
        return
    x = Tensor(
        np.array([[[1, -1, 1], [2, -2, 2]], [[3, -3, 3], [4, -4, 4]], [[5, -5, 5], [6, -6, 6]]]), mstype.float32)
    begin = Tensor(
        np.array([0, 1, 0]), mstype.int64)
    size = Tensor(
        np.array([2, 1, 2]), mstype.int64)
    slice_net = SliceNet()
    output = slice_net(x, begin, size)

    expect = np.array([[[2., -2.]],
                       [[4., -4.]]])
    assert (output.asnumpy() == expect).all()


class GradSlice(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self.grad = GradOperation()
        self.network = network
        self.unique = P.Unique()
        self.reshape = P.Reshape()

    def construct(self, input_x, begin, size):
        return self.grad(self.network)(input_x, begin, size)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_slice_begin_size_tensor_grad():
    """
    /// Feature: Slice op dynamic shape
    /// Description: Slice backward with dynamic shape
    /// Expectation: Euqal to expected value
    """
    if sys.platform != 'linux':
        return
    x = Tensor(
        np.array([[[1, -1, 1], [2, -2, 2]], [[3, -3, 3], [4, -4, 4]], [[5, -5, 5], [6, -6, 6]]]), mstype.float32)
    begin = Tensor(
        np.array([0, 1, 0]), mstype.int64)
    size = Tensor(
        np.array([2, 1, 2]), mstype.int64)

    net = GradSlice(SliceNet())
    output = net(x, begin, size)
    expect = np.array([[[0., 0., 0.],
                        [1., 1., 0.]],

                       [[0., 0., 0.],
                        [1., 1., 0.]],

                       [[0., 0., 0.],
                        [0., 0., 0.]]])
    assert (output.asnumpy() == expect).all()


class ReduceMeanNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.reshape = P.Reshape()

    def construct(self, x, shape):
        y = self.reshape(x, shape)
        return self.reduce_mean(y, 0)


class GradReduceMean(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self.grad = GradOperation(get_all=True)
        self.network = network
        self.reshape = P.Reshape()
        self.unique = P.Unique()

    def construct(self, input_x, shape):
        return self.grad(self.network)(input_x, shape)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_reducemean_dynamic_cpu():
    """
    /// Feature: ReduceMean op dynamic shape
    /// Description: ReduceMean forward with dynamic shape
    /// Expectation: Euqal to expected value
    """
    if sys.platform != 'linux':
        return
    x = Tensor(np.array([10, 10, 2, 2]), mstype.float32)
    x2 = Tensor(np.array([2, 2]), mstype.int64)
    reduce_mean = ReduceMeanNet()
    out = reduce_mean(x, x2)
    expect = np.array([[6., 6.]])
    assert (out.asnumpy() == expect).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_reducemean_dynamic_grad_cpu():
    """
    /// Feature: ReduceMean op dynamic shape
    /// Description: ReduceMean backward with dynamic shape
    /// Expectation: Euqal to expected value
    """
    if sys.platform != 'linux':
        return
    x = Tensor(np.array([10, 10, 2, 2]), mstype.float32)
    x2 = Tensor(np.array([2, 2]), mstype.int64)
    reduce_mean = GradReduceMean(ReduceMeanNet())
    out = reduce_mean(x, x2)
    expect = np.array([[0.5, 0.5, 0.5, 0.5]])
    assert (out[0].asnumpy() == expect).all()
