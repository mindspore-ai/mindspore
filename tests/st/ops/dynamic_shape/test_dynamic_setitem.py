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
from mindspore import context, nn
from mindspore import Tensor
import mindspore.common.dtype as mstype


class NumpySetItem():
    def __init__(self, index, value):
        super(NumpySetItem, self).__init__()
        self.index = index
        self.value = value

    def __call__(self, tensor1, tensor2):
        tensor1[self.index] = self.value
        tensor2[self.index] = self.value
        return tensor1, tensor2


class TensorSetItem(nn.Cell):
    def __init__(self, index, value):
        super(TensorSetItem, self).__init__()
        self.index = index
        self.value = value

    def construct(self, tensor1, tensor2):
        tensor1[self.index] = self.value
        tensor2[self.index] = self.value
        return tensor1, tensor2


def common_func(ms_net, np_net):
    x = Tensor(shape=[8, None, 3], dtype=mstype.float32)
    y = Tensor(shape=[None, 32, 3], dtype=mstype.float32)
    ms_net.set_inputs(x, y)
    input_np1 = np.arange(8 * 16 * 3).reshape(8, 16, 3).astype(np.float32)
    input_np2 = np.arange(16 * 32 * 3).reshape(16, 32, 3).astype(np.float32)
    out0, out1 = ms_net(Tensor(input_np1), Tensor(input_np2))
    out_np0, out_np1 = np_net(input_np1, input_np2)
    assert np.all(out0.asnumpy() == out_np0)
    assert np.all(out1.asnumpy() == out_np1)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_setitem_int_number():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is int, value is a number.
    Expectation: Assert the result is equal the numpy result.
    """
    index = 2
    value = 88.0
    ms_net = TensorSetItem(index, value)
    np_net = NumpySetItem(index, value)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_setitem_int_tensor():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is int, value is a tensor.
    Expectation: Assert the result is equal the numpy result.
    """
    index = 2
    value = Tensor(np.arange(3).reshape(
        (1 * 3)).astype(np.float32), mstype.float32)
    ms_net = TensorSetItem(index, value)
    np_net = NumpySetItem(index, value.asnumpy())
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_setitem_int_sequence():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is int, value is a sequence.
    Expectation: Assert the result is equal the numpy result.
    """
    index = 2
    value = (1.0, Tensor(5, mstype.float32), 8.0)
    ms_net = TensorSetItem(index, value)
    np_net = NumpySetItem(index, value)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_setitem_tensor_number():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is tensor, value is a number.
    Expectation: Assert the result is equal the numpy result.
    """
    index = Tensor(
        np.array([[2, 0, 2], [0, 2, 0], [0, 2, 0]], np.int32), mstype.int32)
    value = 88.0
    ms_net = TensorSetItem(index, value)
    np_net = NumpySetItem(index.asnumpy(), value)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_setitem_tensor_tensor():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is tensor, value is a tensor.
    Expectation: Assert the result is equal the numpy result.
    """
    index = Tensor(
        np.array([[2, 0, 2], [0, 2, 0], [0, 2, 0]], np.int32), mstype.int32)
    value = Tensor(np.arange(3).reshape(
        (1 * 3)).astype(np.float32), mstype.float32)
    ms_net = TensorSetItem(index, value)
    np_net = NumpySetItem(index.asnumpy(), value.asnumpy())
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_setitem_tensor_sequence():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is tensor, value is a sequence.
    Expectation: Assert the result is equal the numpy result.
    """
    index = Tensor(
        np.array([[2, 0, 2], [0, 2, 0], [0, 2, 0]], np.int32), mstype.int32)
    value = (1.0, Tensor(5, mstype.float32), 8.0)
    ms_net = TensorSetItem(index, value)
    np_net = NumpySetItem(index.asnumpy(), value)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_setitem_none_number():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is None, value is a number.
    Expectation: Assert the result is equal the numpy result.
    """
    index = None
    value = 88.0
    ms_net = TensorSetItem(index, value)
    np_net = NumpySetItem(index, value)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_setitem_none_tensor():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is None, value is a tensor.
    Expectation: Assert the result is equal the numpy result.
    """
    index = None
    value = Tensor(np.arange(3).reshape(
        (1 * 3)).astype(np.float32), mstype.float32)
    ms_net = TensorSetItem(index, value)
    np_net = NumpySetItem(index, value.asnumpy())
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_setitem_none_sequence():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is None, value is a sequence.
    Expectation: Assert the result is equal the numpy result.
    """
    index = None
    value = (1.0, Tensor(5, mstype.float32), 8.0)
    ms_net = TensorSetItem(index, value)
    np_net = NumpySetItem(index, value)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_setitem_ellipsis_number():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is ..., value is a number.
    Expectation: Assert the result is equal the numpy result.
    """
    index = ...
    value = 88.0
    ms_net = TensorSetItem(index, value)
    np_net = NumpySetItem(index, value)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_setitem_ellipsis_tensor():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is ..., value is a tensor.
    Expectation: Assert the result is equal the numpy result.
    """
    index = ...
    value = Tensor(np.arange(3).reshape(
        (1 * 3)).astype(np.float32), mstype.float32)
    ms_net = TensorSetItem(index, value)
    np_net = NumpySetItem(index, value.asnumpy())
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_setitem_ellipsis_sequence():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is ..., value is a sequence.
    Expectation: Assert the result is equal the numpy result.
    """
    index = ...
    value = (1.0, Tensor(5, mstype.float32), 8.0)
    ms_net = TensorSetItem(index, value)
    np_net = NumpySetItem(index, value)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_setitem_bool_number():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is bool(True), value is a number.
    Expectation: Assert the result is equal the numpy result.
    """
    index = True
    value = 88.0
    ms_net = TensorSetItem(index, value)
    np_net = NumpySetItem(index, value)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_setitem_bool_tensor():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is bool(True), value is a tensor.
    Expectation: Assert the result is equal the numpy result.
    """
    index = True
    value = Tensor(np.arange(3).reshape(
        (1 * 3)).astype(np.float32), mstype.float32)
    ms_net = TensorSetItem(index, value)
    np_net = NumpySetItem(index, value.asnumpy())
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_setitem_bool_sequence():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is bool(True), value is a sequence.
    Expectation: Assert the result is equal the numpy result.
    """
    index = True
    value = (1.0, Tensor(5, mstype.float32), 8.0)
    ms_net = TensorSetItem(index, value)
    np_net = NumpySetItem(index, value)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_setitem_list_number():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is a list of int, value is a number.
    Expectation: Assert the result is equal the numpy result.
    """
    index = [0, 1]
    value = 88.0
    ms_net = TensorSetItem(index, value)
    np_net = NumpySetItem(index, value)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_setitem_list_tensor():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is al ist of bool and int, value is a tensor.
    Expectation: Assert the result is equal the numpy result.
    """
    index = [True, 5]
    value = Tensor(np.arange(3).reshape(
        (1 * 3)).astype(np.float32), mstype.float32)
    ms_net = TensorSetItem(index, value)
    np_net = NumpySetItem(index, value.asnumpy())
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_setitem_list_sequence():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is a list of int, value is a sequence.
    Expectation: Assert the result is equal the numpy result.
    """
    index = [0, 1]
    value = (1.0, Tensor(5, mstype.float32), 8.0)
    ms_net = TensorSetItem(index, value)
    np_net = NumpySetItem(index, value)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    common_func(ms_net, np_net)
