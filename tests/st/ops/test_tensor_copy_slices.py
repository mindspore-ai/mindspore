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
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import Parameter
import mindspore.common.dtype as mstype
from mindspore.ops.operations import _inner_ops
from mindspore import ops as P
from mindspore.ops import composite as C


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.copy_slices = _inner_ops.TensorCopySlices()

    def construct(self, input_x, update, begin, end, strides):
        return self.copy_slices(input_x, update, begin, end, strides)


class GradNet(nn.Cell):
    def __init__(self):
        super(GradNet, self).__init__()
        self.input_x = Parameter(Tensor([[2, 3, 4, 5], [2, 3, 4, 5]], mstype.float32))
        self.relu = P.ReLU()
        self.copy_slices = _inner_ops.TensorCopySlices()

    def construct(self, input_y):
        output = self.copy_slices(self.input_x, input_y, (0, 0), (1, 1), (1, 1))
        return self.relu(output)


def convert_begin_end_strides_to_slice(begin, end, strides):
    result = []
    for x, y, z in zip(begin, end, strides):
        result.append(slice(x, y, z))
    return tuple(result)


def test_tensor_copy_slices_net(input_shape, update_shape, begin, end, strides, dtype):
    input_np = np.zeros(input_shape, dtype)
    update_np = np.ones(update_shape, dtype)
    input_tensor = Tensor(input_np)
    update = Tensor(update_np)
    net = Net()
    output = net(input_tensor, update, begin, end, strides)
    slices = convert_begin_end_strides_to_slice(begin, end, strides)
    input_np[slices] = update_np
    assert (output.asnumpy() == input_np).all()


def test_tensor_copy_slices_net_many_dtype(input_shape, update_shape, begin, end, strides, dtypes):
    for dtype in dtypes:
        test_tensor_copy_slices_net(input_shape, update_shape, begin, end, strides, dtype)

support_dtype = (np.int64, np.int32, np.float64, np.float32)


def test_tensor_copy_slices():
    test_tensor_copy_slices_net_many_dtype((10,), (5,), (0,), (5,), (1,), support_dtype)
    test_tensor_copy_slices_net_many_dtype((10,), (5,), (5,), (10,), (1,), support_dtype)
    test_tensor_copy_slices_net_many_dtype((10, 10), (5, 10), (0,), (5,), (1,), support_dtype)
    test_tensor_copy_slices_net_many_dtype((10, 10), (5, 10), (5,), (10,), (1,), support_dtype)
    test_tensor_copy_slices_net_many_dtype((10, 10), (5,), (9, 5), (10, 10), (1, 1), support_dtype)
    test_tensor_copy_slices_net_many_dtype((10, 10, 10), (5, 10), (0, 5, 0), (1, 10, 10), (1, 1, 1,), support_dtype)
    test_tensor_copy_slices_net_many_dtype((10, 10, 10), (5, 10), (9, 5,), (10, 10,), (1, 1,), support_dtype)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_copy_slices_bprop():
    input_y = Tensor(2.0, mstype.float32)
    net = GradNet()
    net.set_grad()
    grad_net = C.GradOperation(get_all=True, sens_param=True)(net)

    output = net(input_y)
    grad_output = grad_net(input_y, output)
    assert np.allclose(grad_output[0].asnumpy(), np.array([2.0]))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tensor_copy_slices_ascend_graph():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_tensor_copy_slices()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tensor_copy_slices_ascend_pynative():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    test_tensor_copy_slices()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_copy_slices_gpu_graph():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_tensor_copy_slices()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_copy_slices_gpu_pynative():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    test_tensor_copy_slices()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_copy_slices_cpu_graph():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_tensor_copy_slices()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_copy_slices_cpu_pynative():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_tensor_copy_slices()
