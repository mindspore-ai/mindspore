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
from mindspore import Tensor


class NumpyGetItem():
    def __init__(self, index1, index2):
        super(NumpyGetItem, self).__init__()
        self.index1 = index1
        self.index2 = index2

    def __call__(self, tensor1, tensor2):
        return tensor1[self.index1], tensor2[self.index2]


class TensorGetItem(nn.Cell):
    def __init__(self, index1, index2):
        super(TensorGetItem, self).__init__()
        self.index1 = index1
        self.index2 = index2

    def construct(self, tensor1, tensor2):
        return tensor1[self.index1], tensor2[self.index2]


def common_func(ms_net, np_net):
    x = Tensor(shape=[8, None, 32], dtype=mindspore.float32)
    y = Tensor(shape=[None, 32, 32], dtype=mindspore.float32)
    ms_net.set_inputs(x, y)
    input_np1 = np.arange(8 * 16 * 32).reshape(8, 16, 32).astype(np.float32)
    input_np2 = np.arange(16 * 32 * 32).reshape(16, 32, 32).astype(np.float32)
    out0, out1 = ms_net(Tensor(input_np1), Tensor(input_np2))
    out_np0, out_np1 = np_net(input_np1, input_np2)
    assert np.all(out0.asnumpy() == out_np0)
    assert np.all(out1.asnumpy() == out_np1)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_int_negative():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is negative int.
    Expectation: Assert the result is equal the numpy result.
    """
    index1 = -2
    index2 = -1
    ms_net = TensorGetItem(index1, index2)
    np_net = NumpyGetItem(index1, index2)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_int():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is int.
    Expectation: Assert the result is equal the numpy result.
    """
    index1 = 2
    index2 = 1
    ms_net = TensorGetItem(index1, index2)
    np_net = NumpyGetItem(index1, index2)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_tuple_basic():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is basic tuple.
    Expectation: Assert the result is equal the numpy result.
    """
    index1 = (1, slice(0, 1, 1), ...)
    index2 = (slice(2, None, None), 1, slice(3, 4, None))
    ms_net = TensorGetItem(index1, index2)
    np_net = NumpyGetItem(index1, index2)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_tuple_basic_neg():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is basic tuple(int is negative).
    Expectation: Assert the result is equal the numpy result.
    """
    index1 = (slice(0, 1, 1), ..., -1)
    index2 = (-2, slice(2, None, None), slice(3, 4, None))
    ms_net = TensorGetItem(index1, index2)
    np_net = NumpyGetItem(index1, index2)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_tuple():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is tuple.
    Expectation: Assert the result is equal the numpy result.
    """
    tensor_index = Tensor(np.array([[1, 2, 1], [0, 3, 2]]), mindspore.int32)
    index1 = (slice(2, None, None), (0, 2, 1), tensor_index)
    index2 = (-1, slice(0, 1, None), tensor_index)
    ms_net = TensorGetItem(index1, index2)
    index3 = (slice(2, None, None), (0, 2, 1), tensor_index.asnumpy())
    index4 = (-1, slice(0, 1, None), tensor_index.asnumpy())
    np_net = NumpyGetItem(index3, index4)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_bool():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is bool.
    Expectation: Assert the result is equal the numpy result.
    """
    index = True
    ms_net = TensorGetItem(index, index)
    np_net = NumpyGetItem(index, index)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_none():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is none.
    Expectation: Assert the result is equal the numpy result.
    """
    index = None
    ms_net = TensorGetItem(index, index)
    np_net = NumpyGetItem(index, index)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_ellipsis():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is ellipsis.
    Expectation: Assert the result is equal the numpy result.
    """
    index = ...
    ms_net = TensorGetItem(index, index)
    np_net = NumpyGetItem(index, index)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_slice():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is slice.
    Expectation: Assert the result is equal the numpy result.
    """
    index1 = slice(1, 5, 1)
    index2 = slice(1, None, None)
    ms_net = TensorGetItem(index1, index2)
    np_net = NumpyGetItem(index1, index2)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_slice_neg():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is negative slice.
    Expectation: Assert the result is equal the numpy result.
    """
    index1 = slice(-3, -1, 1)
    index2 = slice(-1, None, None)
    ms_net = TensorGetItem(index1, index2)
    np_net = NumpyGetItem(index1, index2)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_tensor():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is tensor.
    Expectation: Assert the result is equal the numpy result.
    """
    index1 = Tensor(np.array([[1, 2], [0, 3]]), mindspore.int32)
    index2 = Tensor(np.array([[1, 2]]), mindspore.int32)
    ms_net = TensorGetItem(index1, index2)
    np_net = NumpyGetItem(index1.asnumpy(), index2.asnumpy())
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_list():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is list.
    Expectation: Assert the result is equal the numpy result.
    """
    index1 = [True, 2, True]
    index2 = [1, 2, 0]
    ms_net = TensorGetItem(index1, index2)
    np_net = NumpyGetItem(index1, index2)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_slice_startoversize():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is slice and start is over size.
    Expectation: Assert the result is equal the numpy result.
    """
    index1 = slice(8, None, 1)
    index2 = slice(30, None, None)
    ms_net = TensorGetItem(index1, index2)
    np_net = NumpyGetItem(index1, index2)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
