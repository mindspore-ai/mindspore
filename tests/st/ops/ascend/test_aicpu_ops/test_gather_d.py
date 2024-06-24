# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import numpy as np
import pytest

import mindspore
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.functional import vmap
import mindspore.numpy as ms_np


context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, dim=0):
        super(Net, self).__init__()
        self.op = P.GatherD()
        self.dim = dim

    def construct(self, x, index):
        return self.op(x, self.dim, index)


class TensorNet(nn.Cell):
    def __init__(self, dim=0):
        super(TensorNet, self).__init__()
        self.dim = dim

    def construct(self, x, index):
        return x.gather_elements(self.dim, index)


class NetGrad(nn.Cell):
    def __init__(self):
        super(NetGrad, self).__init__()
        self.op = G.GatherDGradV2()

    def construct(self, index, x, dim=0, shape=None):
        return self.op(shape, dim, index, x)


def get_data(ms_type):
    x = Tensor(np.array([[772, 231, 508, 545, 615, 249],
                         [923, 210, 480, 696, 482, 761],
                         [465, 904, 521, 824, 607, 669],
                         [156, 539, 56, 159, 916, 566],
                         [122, 676, 714, 261, 19, 936]]), ms_type)
    dim = 0
    index = Tensor(np.array([[0, 1, 0, 1, 0, -4],
                             [0, 2, 0, 2, 0, -3],
                             [0, 0, 0, 3, 3, -2],
                             [4, 4, 4, 0, 0, -1],
                             [4, 3, 2, 1, -1, -2]]), mindspore.int32)
    expect = np.array([[772, 210, 508, 696, 615, 761],
                       [772, 904, 508, 824, 615, 669],
                       [772, 231, 508, 159, 916, 566],
                       [122, 676, 714, 545, 615, 936],
                       [122, 539, 521, 696, 19, 566]])
    res = (x, dim, index, expect)
    return res


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('ms_type', [mindspore.int32, mindspore.uint32, mindspore.float32])
def test_net(ms_type):
    """
    Feature: test GatherD static shape.
    Description: input x and index is static shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE)
    x, dim, index, expect = get_data(ms_type)
    net = Net(dim)
    out = net(x, index)

    assert np.array_equal(out.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('ms_type', [mindspore.int32, mindspore.uint32, mindspore.float32])
def test_gatherd_dynamic(ms_type):
    """
    Feature: test GatherD dynamic shape.
    Description: index is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE)
    x, dim, index, expect = get_data(ms_type)
    index_dyn = Tensor(shape=[index.shape[0], None], dtype=mindspore.int32)
    net = Net(dim)
    net.set_inputs(x, index_dyn)
    out = net(x, index)

    assert np.array_equal(out.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_functional():
    """
    Feature: test GatherD function interface.
    Description: input x and index is static shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(np.array([[1, 2], [3, 4]]), mindspore.int32)
    index = Tensor(np.array([[0, 0], [1, 0]]), mindspore.int32)
    dim = 1
    output = ops.gather_elements(x, dim, index)
    expect = np.array([[1, 1], [4, 3]])
    assert np.array_equal(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pynative_tensor():
    """
    Feature: test GatherD tensor interface in pynative case.
    Description: input x and index is static shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = ms_np.array([[1, 2], [3, 4]])
    dim = 1
    index = ms_np.array([[0, 0], [1, 0]])
    output = x.gather_elements(dim, index)
    expect = np.array([[1, 1], [4, 3]])
    assert np.array_equal(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_graph_tensor():
    """
    Feature: test GatherD tensor interface in graph case.
    Description: input x and index is static shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = ms_np.array([[1, 2], [3, 4]])
    index = ms_np.array([[0, 0], [1, 0]])
    dim = 1
    net = TensorNet(dim)
    output = net(x, index)
    expect = np.array([[1, 1], [4, 3]])
    assert np.array_equal(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype', [np.int32, np.uint32, np.float32])
def test_gatherd_vmap(dtype):
    """
    Feature: test GatherD vmap interface.
    Description: input x and index is static shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    def cal_gatherd(x, dim, index):
        return P.GatherD()(x, dim, index)

    gather_dim = 1
    x = Tensor(np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]]).astype(dtype))
    y = Tensor(np.array([[[0, 0], [1, 0]], [[0, 0], [1, 0]], [[0, 0], [1, 0]]]).astype(np.int32))
    outputs = vmap(cal_gatherd, in_axes=(0, None, 0), out_axes=0)(x, gather_dim, y)
    expect = np.array([[[1, 1], [4, 3]], [[1, 1], [4, 3]], [[1, 1], [4, 3]]]).astype(dtype)
    assert np.allclose(outputs.asnumpy(), expect)


def test_net_bool():
    x = Tensor(np.array([[0, 1, 0, 0, 1, 0],
                         [0, 1, 0, 0, 1, 0],
                         [0, 0, 1, 1, 0, 1],
                         [1, 0, 1, 1, 0, 0],
                         [1, 1, 1, 1, 0, 0]]), mindspore.bool_)
    index = Tensor(np.array([[0, 0, 0, 1, 1],
                             [0, 0, 0, 1, 4],
                             [0, 0, 0, 1, -1],
                             [1, 1, 1, 0, 0]]), mindspore.int32)
    dim = 0
    net = Net(dim)
    out = net(x, index)
    print(out.asnumpy())

    expect_out = np.array([[0, 1, 0, 0, 1],
                           [0, 1, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 1, 0, 0, 1]]).astype(np.bool)
    assert np.array_equal(out.asnumpy(), expect_out)


def test_net_grad():
    index = Tensor(np.array([[0, 1, 2, 0, 0],
                             [2, 0, 0, 1, -1]]), mindspore.int32)
    x = Tensor(np.array([[772, 231, 508, 615, 249],
                         [122, 676, 714, 261, 936]]), mindspore.int32)
    net = NetGrad()
    shape = Tensor(np.array([3, 5]), mindspore.int64)
    dim = 0
    out = net(shape, dim, index, x)
    print(out.asnumpy())

    expect_out = np.array([[772, 676, 714, 615, 249],
                           [0, 231, 0, 261, 0],
                           [122, 0, 508, 0, 936]])
    assert np.array_equal(out.asnumpy(), expect_out)
