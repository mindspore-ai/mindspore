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


class TileNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.tile = P.Tile()

    def construct(self, x, multiples):
        out = self.tile(x, multiples)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tile_multiple_tensor_cpu():
    """
    /// Feature: Tile op dynamic shape
    /// Description: Tile forward with dynamic shape
    /// Expectation: Euqal to expected value
    """
    if sys.platform != 'linux':
        return
    multiples_1 = (2, 1)
    multiples_2 = (4, 1)
    x = Tensor(np.array([[1, 2, 3, 4]]), mstype.float32)
    tile_net = TileNet()
    expect_1 = np.array([[1., 2., 3., 4.],
                         [1., 2., 3., 4.]])
    expect_2 = np.array([[1., 2., 3., 4.],
                         [1., 2., 3., 4.],
                         [1., 2., 3., 4.],
                         [1., 2., 3., 4.]])
    expect = [expect_1, expect_2]
    for i, multiples in enumerate([multiples_1, multiples_2]):
        output = tile_net(x, multiples)
        assert (output.asnumpy() == expect[i]).all()


class GradTile(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self.grad = GradOperation(sens_param=True)
        self.network = network
        self.unique = P.Unique()
        self.reshape = P.Reshape()

    def construct(self, input_x, multiples, grad):
        dy = self.unique(grad)[0]
        dy = self.reshape(dy, (2, 4))
        return self.grad(self.network)(input_x, multiples, dy)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tile_multiple_tensor_grad_cpu():
    """
    /// Feature: Tile op dynamic shape
    /// Description: Tile backward with dynamic shape
    /// Expectation: Euqal to expected value
    """
    if sys.platform != 'linux':
        return
    multiples = Tensor(np.array([2, 1]), mstype.int64)
    x0 = Tensor(np.array([[1, 2, 3, 4]]), mstype.float32)
    tile_net = GradTile(TileNet())
    dout = Tensor(np.arange(1, 9), mstype.float32)
    output = tile_net(x0, multiples, dout)
    expect = np.array([[6., 8., 10., 12.]])
    assert (output.asnumpy() == expect).all()


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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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


class ConcatNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.unique = P.Unique()
        self.concat = P.Concat()
        self.reshape = P.Reshape()

    def construct(self, x, y, z, shape_tensor):
        x = self.reshape(x, shape_tensor)
        y = self.reshape(y, shape_tensor)
        z = self.reshape(z, shape_tensor)
        out = self.concat((x, y, z))
        return out


class GradConcat(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self.grad = GradOperation(sens_param=True)
        self.network = network
        self.unique = P.Unique()
        self.reshape = P.Reshape()

    def construct(self, x, y, z, shape, grad):
        # grad = self.reshape(grad, (-1,))
        dy = self.reshape(self.unique(grad)[0], (-1, 1, 2, 1))
        return self.grad(self.network)(x, y, z, shape, dy)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_dynamic_grad_cpu():
    """
    /// Feature: Concat op dynamic shape
    /// Description: Concat backward with dynamic shape
    /// Expectation: Euqal to expected value
    """
    if sys.platform != 'linux':
        return
    x = Tensor(np.array([1, 2, 3, 4, 5, 6]), mstype.float32)
    x2 = Tensor(np.array([1, 2, 3, 4, 5, 6]), mstype.float32)
    x3 = Tensor(np.array([1, 2, 3, 4, 5, 6]), mstype.float32)
    shape = Tensor(np.array([3, 1, 2, 1]), mstype.int64)
    dout = Tensor(np.arange(1, 19), mstype.float32)
    net = GradConcat(ConcatNet())
    output = net(x, x2, x3, shape, dout)
    expect = np.array([1., 2., 3., 4., 5., 6.])
    assert (output.asnumpy() == expect).all()


class SliceNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.slice = P.Slice()

    def construct(self, x, begin, size):
        return self.slice(x, begin, size)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
        self.grad = GradOperation(sens_param=True)
        self.network = network
        self.unique = P.Unique()
        self.reshape = P.Reshape()

    def construct(self, input_x, begin, size, grad):
        # grad = self.reshape(grad, (-1,))
        dy = self.unique(grad)[0]
        dy = self.reshape(dy, size)
        return self.grad(self.network)(input_x, begin, size, dy)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_slice_begin_size_tensor_grad():
    """
    /// Feature: Slice op dynamic shape
    /// Description: Slice backward with dynamic shape
    /// Expectation: Euqal to expected value
    """
    if sys.platform != 'linux':
        return
    dy = Tensor(np.array([1, 2, 3, 4]), mstype.float32)
    x = Tensor(
        np.array([[[1, -1, 1], [2, -2, 2]], [[3, -3, 3], [4, -4, 4]], [[5, -5, 5], [6, -6, 6]]]), mstype.float32)
    begin = Tensor(
        np.array([0, 1, 0]), mstype.int64)
    size = Tensor(
        np.array([2, 1, 2]), mstype.int64)

    net = GradSlice(SliceNet())
    output = net(x, begin, size, dy)
    expect = np.array([[[0., 0., 0.],
                        [1., 2., 0.]],

                       [[0., 0., 0.],
                        [3., 4., 0.]],

                       [[0., 0., 0.],
                        [0., 0., 0.]]])
    assert (output.asnumpy() == expect).all()


class ReduceMeanNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.reshape = P.Reshape()
        self.tile = P.Tile()

    def construct(self, x, shape):
        y = self.reshape(x, shape)
        return self.reduce_mean(y, 0)


class GradReduceMean(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network
        self.reshape = P.Reshape()
        self.unique = P.Unique()

    def construct(self, input_x, shape, grad):
        grad = self.reshape(self.unique(grad)[0], (1, 2))
        return self.grad(self.network)(input_x, shape, grad)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
    dout = Tensor(np.array([1, 3]), mstype.float32)
    reduce_mean = GradReduceMean(ReduceMeanNet())
    out = reduce_mean(x, x2, dout)
    expect = np.array([[0.5, 1.5, 0.5, 1.5]])
    assert (out[0].asnumpy() == expect).all()
