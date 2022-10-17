# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops.functional import vmap
from mindspore.ops.operations import _inner_ops as inner


class BatchToSpaceNDNet(nn.Cell):
    def __init__(self, nptype, block_shape=2, input_shape=(4, 1, 1, 1)):
        super(BatchToSpaceNDNet, self).__init__()
        self.batch_to_space_nd = ops.BatchToSpaceND(block_shape=block_shape, crops=[[0, 0], [0, 0]])
        input_size = np.prod(input_shape)
        data_np = np.arange(input_size).reshape(input_shape).astype(nptype)
        self.x1 = Parameter(initializer(Tensor(data_np), input_shape), name='x1')

    @ms_function
    def construct(self):
        y1 = self.batch_to_space_nd(self.x1)
        return y1


def batch_to_space_nd_test_case(nptype, block_shape=2, input_shape=(4, 1, 1, 1)):
    expect = np.array([[[[0, 1],
                         [2, 3]]]]).astype(nptype)

    dts = BatchToSpaceNDNet(nptype, block_shape, input_shape)
    output = dts()

    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float16, np.int8, np.int32, np.uint8, np.uint32])
def test_batch_to_space_nd_graph(dtype):
    """
    Feature: test BatchToSpaceND function interface.
    Description: test interface.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    batch_to_space_nd_test_case(dtype)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float16, np.int8, np.int32, np.uint8, np.uint32])
def test_batch_to_space_nd_pynative(dtype):
    """
    Feature: test BatchToSpaceND function interface.
    Description: test interface.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    batch_to_space_nd_test_case(dtype)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batch_to_space_nd_function():
    """
    Feature: test BatchToSpaceND function interface.
    Description: test interface.
    Expectation: the result match with numpy result
    """
    context.set_context(device_target="CPU")
    x = Tensor(np.arange(4).reshape((4, 1, 1, 1)).astype(np.float32), mindspore.float32)
    output = ops.batch_to_space_nd(x, 2, [[0, 0], [0, 0]])
    expect = np.array([[[[0, 1],
                         [2, 3]]]]).astype(np.float32)
    np.testing.assert_array_equal(output.asnumpy(), expect)


class BatchToSpaceNDDynamicShapeNetMS(nn.Cell):
    def __init__(self, block_shape, crops, is_dynamic_rank):
        super().__init__()
        self.batch_to_space_nd = ops.BatchToSpaceND(block_shape, crops)
        self.convert_to_dynamic = inner.ConvertToDynamic(
            is_dynamic_rank=is_dynamic_rank).add_prim_attr("primitive_target", "CPU")

    def construct(self, x):
        x = self.convert_to_dynamic(x)
        return self.batch_to_space_nd(x)


def batch_to_space_nd_dynamic(is_dynamic_rank):
    """
    Feature: test BatchToSpaceND dynamic shape.
    Description: the input to BatchToSpaceND is dynamic.
    Expectation: the result match with numpy result
    """
    x = np.arange(4).reshape((4, 1, 1, 1)).astype(np.float32)
    block_shape = [2, 2]
    crops = [[0, 0], [0, 0]]

    input_x = Tensor(x, mindspore.float32)
    x_dyn = Tensor(shape=[None, None, None, None], dtype=mindspore.float32)
    expect = np.array([[[[0, 1],
                         [2, 3]]]]).astype(np.float32)
    dyn_net = BatchToSpaceNDDynamicShapeNetMS(block_shape, crops, is_dynamic_rank)
    dyn_net.set_inputs(x_dyn)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    output = dyn_net(input_x)
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    output = dyn_net(input_x)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batch_to_space_nd_dynamic_case():
    """
    Feature: test BatchToSpaceND dynamic shape.
    Description: the input to BatchToSpaceND is dynamic.
    Expectation: the result match with numpy result
    """
    batch_to_space_nd_dynamic(True)
    batch_to_space_nd_dynamic(False)


def vmap_case():
    class Net(nn.Cell):
        def __init__(self, block_shape, crops):
            super(Net, self).__init__()
            self.batch_to_space_nd = ops.BatchToSpaceND(block_shape, crops)

        def construct(self, a):
            return self.batch_to_space_nd(a)

    class WrapNet(nn.Cell):
        def __init__(self, net, in_axes, out_axes):
            super(WrapNet, self).__init__()
            self.net = net
            self.in_axes = in_axes
            self.out_axes = out_axes

        def construct(self, input_x):
            return vmap(self.net, self.in_axes, self.out_axes)(input_x)

    block_shape = [2, 2]
    crops = [[0, 0], [0, 0]]
    input_shape = (2, 4, 1, 1, 1)
    data_np = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
    net = Net(block_shape, crops)

    # test input axis and output axis are the same
    v_net_1 = WrapNet(Net(block_shape, crops), (0,), 0)
    output_v = v_net_1(Tensor(data_np)).asnumpy()

    for i in range(input_shape[0]):
        assert np.allclose(output_v[i, :, :, :, :], net(Tensor(data_np[i, :, :, :, :])).asnumpy())

    # test input axis and output axis are different
    v_net_2 = WrapNet(Net(block_shape, crops), (0,), 1)
    output_v = v_net_2(Tensor(data_np)).asnumpy()

    for i in range(input_shape[0]):
        assert np.allclose(output_v[:, i, :, :, :], net(Tensor(data_np[i, :, :, :, :])).asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batch_to_space_nd_vmap_cpu():
    """
    Feature: test SpactToBatchND vmap on CPU.
    Description: inputs with batch.
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    vmap_case()
