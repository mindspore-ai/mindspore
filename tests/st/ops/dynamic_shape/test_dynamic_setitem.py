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
from tests.mark_utils import arg_mark
import numpy as np
import pytest
from mindspore import context, nn, ops, jit
from mindspore import Tensor, ParameterTuple, mutable
from mindspore.ops.composite import GradOperation
from mindspore.nn import Cell
import mindspore.common.dtype as mstype

context.set_context(mode=context.GRAPH_MODE)


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


class GradOfFirstInput(_Grad):
    """
    get grad of first input
    """

    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(sens_param=sens_param),
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


class DynamicRankCommonFunc():
    def __init__(self, ms_net, np_net, input_np, axis_np):
        super().__init__()
        self.ms_net = ms_net
        self.input_np_t = Tensor(input_np)
        self.axis_np_t = Tensor(axis_np)
        axis_dyn = Tensor(shape=(None,), dtype=self.axis_np_t.dtype)
        self.ms_net.set_inputs(self.input_np_t, axis_dyn)
        self.ms_net.set_grad()
        self.np_net = np_net

        self.input_np = input_np
        self.axis_np = axis_np

        self.out_np = np.array(1).astype(input_np.dtype)

    def forward_cmp(self):
        out_ms = self.ms_net(self.input_np_t, self.axis_np_t)
        self.out_np = self.np_net(self.input_np, self.axis_np)
        assert np.allclose(out_ms.asnumpy(), self.out_np, rtol=0.0001)

    def grad_impl(self):
        grad_net = GradOfFirstInput(self.ms_net)
        grad_net.set_train()
        grad_net(self.input_np_t, self.axis_np_t, Tensor(self.out_np))


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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
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
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
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
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
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
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
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
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
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
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
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
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
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
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
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
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
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
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
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
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
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
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
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
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_setitem_slice_sequence():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is a slice, value is a sequence.
    Expectation: Assert the result is equal the numpy result.
    """
    index = slice(0, None, 2)
    value = (1.0, Tensor(5, mstype.float32), 8.0)
    ms_net = TensorSetItem(index, value)
    np_net = NumpySetItem(index, value)
    fact = CommonFunc(ms_net, np_net)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_with_single_basic_index():
    """
    Feature: Test Tensor slice for dynamic rank in feed mode.
    Description: The input shape is dynamic and the tensor index is ellipsis/None/Integer/Slice/Bool.
    Expectation: Assert the result is equal the numpy result.
    """
    class TensorDynamciSetItem(Cell):
        def __init__(self):
            super().__init__()
            self.extra = Tensor(0)

        def construct(self, x, axis):
            x = ops.reduce_sum(x, axis)
            x[...] = 1
            x[False] = 1
            x[None] = 1
            x[1:4:2] = 1
            x[x.shape[0]-3:ops.dyn_shape(x)[0]:Tensor(2)] = 1
            x[True] = 1
            x[()] = 1
            return x

    class NpSetItem():
        @classmethod
        def __call__(cls, x, axis):
            x = x.sum(axis=axis[0])
            x[...] = 1
            x[False] = 1
            x[None] = 1
            x[1:4:2] = 1
            x[x.shape[0]-3:4:2] = 1
            x[True] = 1
            x[()] = 1
            return x

    input_np = np.random.randn(3, 6, 4, 4, 3).astype(np.float32)
    axis_np = np.array([0])
    ms_net = TensorDynamciSetItem()
    np_net = NpSetItem()
    context.set_context(mode=context.GRAPH_MODE)
    fact = DynamicRankCommonFunc(ms_net, np_net, input_np, axis_np)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.skip(reason="Need to be fixed.")
@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_tuple_with_basic_index():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic and the tensor index is tuple
     (integer, slice, ellipsis ,None).
    Expectation: Assert the result is equal the numpy result.
    """

    class TensorDynamciSetItem(Cell):
        def __init__(self):
            super().__init__()
            self.extra = Tensor(0)

        def construct(self, x, axis):
            x = ops.reduce_sum(x, axis)
            x_shape = x.shape[0]
            x[1:x_shape:2, 1:x_shape:2, ..., x_shape -
              3, None] = Tensor([1], mstype.float32)
            x[..., 1:x_shape:1, 1:x_shape:1, x_shape -
              2, None] = Tensor([1], mstype.float32)
            x[1:x_shape:2, 1:x_shape:2, x_shape-2,
              None, ...] = Tensor([1], mstype.float32)
            return x

    class NpSetItem():
        @classmethod
        def __call__(cls, x, axis):
            x = x.sum(axis=axis[0])
            x_shape = x.shape[0]
            x[1:x_shape:2, 1:x_shape:2, ..., x_shape-3, None] = 1
            x[..., 1:x_shape:1, 1:x_shape:1, x_shape-2, None] = 1
            x[1:x_shape:2, 1:x_shape:2, x_shape-2, None, ...] = 1
            return x

    input_np = np.random.randn(3, 4, 5, 6, 7).astype(np.float32)
    axis_np = np.array([0])
    ms_net = TensorDynamciSetItem()
    np_net = NpSetItem()
    context.set_context(mode=context.GRAPH_MODE)
    fact = DynamicRankCommonFunc(ms_net, np_net, input_np, axis_np)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.skip(reason="Need to be fixed.")
@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_tuple_with_mix_index():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic and the tensor index is tuple
     (integer, slice, ellipsis, tensor, bool ,list).
    Expectation: Assert the result is equal the numpy result.
    """

    class TensorDynamciSetItem(Cell):
        def __init__(self):
            super().__init__()
            self.extra = Tensor(0)

        def construct(self, x, axis):
            x = ops.reduce_sum(x, axis)
            x[Tensor(1), 1, [1, 2], None, ...] = Tensor([1], mstype.float32)
            x[..., Tensor(1), 1, [1, 2], None] = Tensor([1], mstype.float32)
            x[Tensor(1), 1, ..., [1, 2], None] = Tensor([1], mstype.float32)
            return x

    class NpSetItem():
        @classmethod
        def __call__(cls, x, axis):
            x = x.sum(axis=axis[0])
            x[np.array(1), 1, [1, 2], None, ...] = 1
            x[..., np.array(1), 1, [1, 2], None] = 1
            x[np.array(1), 1, ..., [1, 2], None] = 1
            return x

    input_np = np.random.randn(3, 6, 4, 4, 3).astype(np.float32)
    axis_np = np.array([0])
    ms_net = TensorDynamciSetItem()
    np_net = NpSetItem()
    context.set_context(mode=context.PYNATIVE_MODE)
    fact = DynamicRankCommonFunc(ms_net, np_net, input_np, axis_np)
    fact.forward_cmp()
    context.set_context(mode=context.GRAPH_MODE)
    fact = DynamicRankCommonFunc(ms_net, np_net, input_np, axis_np)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_tuple_with_multi_tensor_index():
    """
    Feature: Test Tensor slice for dynamic rank in feed mode.
    Description: The input shape is dynamic and the tensor index is multi tensors.
    Expectation: Assert the result is equal the numpy result.
    """
    class Net(Cell):
        def construct(self, x, axis):
            x = ops.reduce_min(x, axis)
            x[Tensor(np.ones((25), int)), :, Tensor(
                np.ones((5, 5)).astype(np.bool))] = Tensor([1], mstype.int64)
            return x

    class NumpyNet():
        @classmethod
        def __call__(cls, x, axis):
            x = x.min(axis=axis[0])
            x[np.ones((25), int), :, np.ones((5, 5)).astype(np.bool)] = 1
            return x

    net_ms = Net()
    net_np = NumpyNet()
    input_np = np.ones((3, 6, 5, 5, 5)).astype(np.int64)
    axis_np = np.array([0])

    context.set_context(mode=context.PYNATIVE_MODE)
    fact = DynamicRankCommonFunc(net_ms, net_np, input_np, axis_np)
    fact.forward_cmp()
    context.set_context(mode=context.GRAPH_MODE)
    fact = DynamicRankCommonFunc(net_ms, net_np, input_np, axis_np)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_tuple_with_empty_bool_tensor_index():
    """
    Feature: Test Tensor slice for dynamic rank in feed mode.
    Description: The input shape is dynamic and the tensor index is multi tensors.
    Expectation: Assert the result is equal the numpy result.
    """
    class Net(Cell):
        def construct(self, x, axis):
            x = ops.reduce_min(x, axis)
            x[:, :, Tensor(np.zeros((5, 5)).astype(np.bool))] = 1
            return x

    class NumpyNet():
        @classmethod
        def __call__(cls, x, axis):
            x = x.min(axis=axis[0])
            x[:, :, np.zeros((5, 5)).astype(np.bool)] = 1
            return x

    net_ms = Net()
    net_np = NumpyNet()
    input_np = np.ones((3, 6, 5, 5, 5)).astype(np.int64)
    axis_np = np.array([0])

    context.set_context(mode=context.PYNATIVE_MODE)
    fact = DynamicRankCommonFunc(net_ms, net_np, input_np, axis_np)
    fact.forward_cmp()
    context.set_context(mode=context.GRAPH_MODE)
    fact = DynamicRankCommonFunc(net_ms, net_np, input_np, axis_np)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_with_list_index():
    """
    Feature: Test Tensor slice for dynamic rank in feed mode.
    Description: The input shape is dynamic and the tensor index is List.
    Expectation: Assert the result is equal the numpy result.
    """
    class Net(Cell):
        def construct(self, x, axis):
            x = ops.reduce_min(x, axis)
            value = mutable([1])
            x[[1, 2]] = value
            return x

    class NumpyNet():
        @classmethod
        def __call__(cls, x, axis):
            x = x.min(axis=axis[0]).min(axis=axis[0])
            x[[1, 2]] = [1]
            return x

    net_ms = Net()
    net_np = NumpyNet()
    input_np = np.ones((3, 3, 3, 4)).astype(np.int64)
    axis_np = np.array([0, 1])

    context.set_context(mode=context.GRAPH_MODE)
    fact = DynamicRankCommonFunc(net_ms, net_np, input_np, axis_np)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_rank_setitem_slice_int():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is a slice, value is a int.
    Expectation: Assert the result is equal the numpy result.
    """
    class TensorDynamciSetItem(Cell):
        def construct(self, x, axis):
            x = ops.reduce_sum(x, axis)
            x[2:None] = 1
            return x

    class NpSetItem():
        @classmethod
        def __call__(cls, x, axis):
            x = x.sum(axis=axis[0]).sum(axis=axis[0])
            x[2:None] = 1
            return x
    input_np = np.random.randn(3, 6, 4).astype(np.float32)
    axis_np = np.array([0, 1])
    ms_net = TensorDynamciSetItem()
    np_net = NpSetItem()
    fact = DynamicRankCommonFunc(ms_net, np_net, input_np, axis_np)
    fact.forward_cmp()
    fact.grad_impl()


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_setitem_with_tensor_index_tensor_value():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is a slice, value is a int.
    Expectation: Assert the result is equal the numpy result.
    """
    @jit
    def foo(data, index):
        data[index] = Tensor([5])
        return data

    data = [Tensor([1]), Tensor([2]), Tensor([3])]
    ret = foo(data, Tensor([0]))
    assert ret == [Tensor([5]), Tensor([2]), Tensor([3])]


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_setitem_with_tensor_index_tensor_value_2():
    """
    Feature: Test index value assignment for dynamic shape Tensor in feed mode.
    Description: The input shape is dynamic, the tensor index is a slice, value is a int.
    Expectation: Assert the result is equal the numpy result.
    """
    @jit
    def foo(data, index, value):
        data[index] = value
        return data

    data = [Tensor([1]), Tensor([2]), Tensor([3])]
    ret = foo(data, Tensor([0]), Tensor([5]))
    assert ret == [Tensor([5]), Tensor([2]), Tensor([3])]
