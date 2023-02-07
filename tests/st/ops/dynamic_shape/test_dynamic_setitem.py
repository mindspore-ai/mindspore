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
from mindspore import Tensor, ParameterTuple
from mindspore.ops.composite import GradOperation
from mindspore.nn import Cell
import mindspore.common.dtype as mstype


class _Grad(Cell):
    def __init__(self, grad, network, wrt_params=False, real_inputs_count=None):
        super().__init__()
        self.network = network
        self.grad = grad
        self.sens_param = self.grad.sens_param
        self.wrt_params = wrt_params
        self.real_inputs_count = real_inputs_count
        if self.wrt_params:
            self.params = ParameterTuple(self.network.trainable_params())

    def construct(self, *inputs):
        if self.real_inputs_count is None or self.sens_param is False:
            if self.wrt_params:
                return self.grad(self.network, self.params)(*inputs)
            return self.grad(self.network)(*inputs)

        real_inputs = inputs[:self.real_inputs_count]
        sense_param_inputs = inputs[self.real_inputs_count:]
        if self.wrt_params:
            return self.grad(self.network, self.params)(*real_inputs, sense_param_inputs)
        return self.grad(self.network)(*real_inputs, sense_param_inputs)


class GradOfAllInputs(_Grad):
    """
    get grad of all inputs
    """

    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(get_all=True, sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


class CommonFunc():
    def __init__(self, ms_net, np_net):
        super(CommonFunc, self).__init__()
        self.ms_net = ms_net
        self.ms_net.set_grad()
        self.np_net = np_net

        input_dyn0 = Tensor(shape=[8, None, 3], dtype=mstype.float32)
        input_dyn1 = Tensor(shape=[None, 32, 3], dtype=mstype.float32)
        ms_net.set_inputs(input_dyn0, input_dyn1)

        self.input_np0 = np.arange(
            8 * 16 * 3).reshape(8, 16, 3).astype(np.float32)
        self.input_np1 = np.arange(
            16 * 32 * 3).reshape(16, 32, 3).astype(np.float32)
        self.input_np0_t = Tensor(self.input_np0)
        self.input_np1_t = Tensor(self.input_np1)
        self.out_np0 = np.array(1).astype(self.input_np0.dtype)
        self.out_np1 = np.array(1).astype(self.input_np1.dtype)

    def forward_cmp(self):
        out_ms0, out_ms1 = self.ms_net(
            self.input_np0_t, self.input_np1_t)
        self.out_np0, self.out_np1 = self. np_net(
            self.input_np0, self.input_np1)
        assert np.all(out_ms0.asnumpy() == self.out_np0)
        assert np.all(out_ms1.asnumpy() == self.out_np1)

    def grad_impl(self):
        grad_net = GradOfAllInputs(self.ms_net)
        grad_net.set_train()
        grad_net(self.input_np0_t, self.input_np1_t,
                 (Tensor(self.out_np0), Tensor(self.out_np1)))


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


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
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
    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
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
    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
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
    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
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
    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
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
    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
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
    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
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
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
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
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
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
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
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
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
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
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
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
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
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
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
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
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
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
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
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
    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
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
    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
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
    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()
