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
import numpy as np
import pytest
import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops.functional import vmap


class SpaceToBatchNDNet(nn.Cell):
    def __init__(self, nptype, block_size=2, input_shape=(1, 1, 4, 4)):
        super(SpaceToBatchNDNet, self).__init__()
        self.space_to_batch_nd = ops.SpaceToBatchND(block_shape=block_size, paddings=[[0, 0], [0, 0]])
        input_size = np.prod(input_shape)
        data_np = np.arange(input_size).reshape(input_shape).astype(nptype)
        self.x1 = Parameter(initializer(Tensor(data_np), input_shape), name='x1')

    @jit
    def construct(self):
        y1 = self.space_to_batch_nd(self.x1)
        return y1


def space_to_batch_nd_test_case(nptype, block_size=2, input_shape=(1, 1, 4, 4)):
    expect = np.array([[[[0, 2],
                         [8, 10]]],
                       [[[1, 3],
                         [9, 11]]],
                       [[[4, 6],
                         [12, 14]]],
                       [[[5, 7],
                         [13, 15]]]]).astype(nptype)

    dts = SpaceToBatchNDNet(nptype, block_size, input_shape)
    output = dts()

    assert (output.asnumpy() == expect).all()


def space_to_batch_nd_all_dtype():
    space_to_batch_nd_test_case(np.float32)
    space_to_batch_nd_test_case(np.float16)
    space_to_batch_nd_test_case(np.int8)
    space_to_batch_nd_test_case(np.int16)
    space_to_batch_nd_test_case(np.int32)
    space_to_batch_nd_test_case(np.int64)
    space_to_batch_nd_test_case(np.uint8)
    space_to_batch_nd_test_case(np.uint16)
    space_to_batch_nd_test_case(np.uint32)
    space_to_batch_nd_test_case(np.uint64)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_space_to_batch_nd_graph():
    """
    Feature: test SpaceToBatchND function interface.
    Description: test interface.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    space_to_batch_nd_all_dtype()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_space_to_batch_nd_pynative():
    """
    Feature: test SpaceToBatchND function interface.
    Description: test interface.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    space_to_batch_nd_all_dtype()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_space_to_batch_nd_function():
    """
    Feature: test SpaceToBatchND function interface.
    Description: test interface.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    x = Tensor(np.arange(16).reshape((1, 1, 4, 4)).astype(np.float32), mindspore.float32)
    output = ops.space_to_batch_nd(x, 2, [[0, 0], [0, 0]])
    expect = np.array([[[[0, 2],
                         [8, 10]]],
                       [[[1, 3],
                         [9, 11]]],
                       [[[4, 6],
                         [12, 14]]],
                       [[[5, 7],
                         [13, 15]]]]).astype(np.float32)
    np.testing.assert_array_equal(output.asnumpy(), expect)


class SpaceToBatchNDDynamicShapeNetMS(nn.Cell):
    def __init__(self, block_size, paddings, axis=0):
        super().__init__()
        self.unique = ops.Unique()
        self.gather = ops.Gather()
        self.space_to_batch_nd = ops.SpaceToBatchND(block_size, paddings)
        self.axis = axis

    def construct(self, x, indices):
        unique_indices, _ = self.unique(indices)
        x = self.gather(x, unique_indices, self.axis)
        return self.space_to_batch_nd(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_space_to_batch_nd_dynamic():
    """
    Feature: test SpaceToBatchND dynamic shape.
    Description: the input to SpaceToBatchND is dynamic.
    Expectation: the result match with numpy result
    """
    x = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8]]], [[[1, 2, 3, 4], [5, 6, 7, 8]]],
                  [[[1, 2, 3, 4], [5, 6, 7, 8]]], [[[1, 2, 3, 4], [5, 6, 7, 8]]]]).astype(np.float32)
    block_size = [2, 2]
    paddings = [[0, 0], [0, 0]]

    input_x = Tensor(x, mindspore.float32)
    input_y = Tensor(np.array([0, 0, 1, 0]), mindspore.int32)
    expect = np.array([[[[1., 3.]]],
                       [[[1., 3.]]],
                       [[[2., 4.]]],
                       [[[2., 4.]]],
                       [[[5., 7.]]],
                       [[[5., 7.]]],
                       [[[6., 8.]]],
                       [[[6., 8.]]]]).astype(np.float32)
    dyn_net = SpaceToBatchNDDynamicShapeNetMS(block_size, paddings)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    output = dyn_net(input_x, input_y)
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    output = dyn_net(input_x, input_y)
    assert (output.asnumpy() == expect).all()


def vmap_case():
    class Net(nn.Cell):
        def __init__(self, block_size, paddings):
            super(Net, self).__init__()
            self.space_to_batch_nd = ops.SpaceToBatchND(block_size, paddings)

        def construct(self, a):
            return self.space_to_batch_nd(a)

    class WrapNet(nn.Cell):
        def __init__(self, net, in_axes, out_axes):
            super(WrapNet, self).__init__()
            self.net = net
            self.in_axes = in_axes
            self.out_axes = out_axes

        def construct(self, input_x):
            return vmap(self.net, self.in_axes, self.out_axes)(input_x)

    block_size = [2, 2]
    paddings = [[0, 0], [0, 0]]
    input_shape = (2, 3, 1, 4, 4)
    data_np = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
    net = Net(block_size, paddings)

    # test input axis and output axis are the same
    v_net_1 = WrapNet(Net(block_size, paddings), (0,), 0)
    output_v = v_net_1(Tensor(data_np)).asnumpy()

    for i in range(input_shape[0]):
        assert np.allclose(output_v[i, :, :, :, :], net(Tensor(data_np[i, :, :, :, :])).asnumpy())

    # test input axis and output axis are different
    v_net_2 = WrapNet(Net(block_size, paddings), (0,), 1)
    output_v = v_net_2(Tensor(data_np)).asnumpy()

    for i in range(input_shape[0]):
        assert np.allclose(output_v[:, i, :, :, :], net(Tensor(data_np[i, :, :, :, :])).asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_space_to_batch_nd_vmap_cpu():
    """
    Feature: test SpactToBatchND vmap on CPU.
    Description: inputs with batch.
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    vmap_case()
