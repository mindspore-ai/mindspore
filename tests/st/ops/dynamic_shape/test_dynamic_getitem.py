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
import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore import Tensor, ops, ParameterTuple
from mindspore.ops.composite import GradOperation
from mindspore.nn import Cell


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


class GradOfFirstInput(_Grad):
    """
    get grad of first input
    """

    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


class CommonFunc():
    def __init__(self, ms_net, np_net, input_np, input_dyn):
        super(CommonFunc, self).__init__()
        self.ms_net = ms_net
        self.ms_net.set_inputs(input_dyn)
        self.ms_net.set_grad()
        self.np_net = np_net

        self.input_np = input_np
        self.input_np_t = Tensor(input_np)
        self.out_np = np.array(1).astype(input_np.dtype)

    def forward_cmp(self):
        out_ms = self.ms_net(self.input_np_t)
        self.out_np = self.np_net(self.input_np)
        assert np.all(out_ms.asnumpy() == self.out_np)

    def grad_impl(self):
        grad_net = GradOfFirstInput(self.ms_net)
        grad_net.set_train()
        grad_net(self.input_np_t, Tensor(self.out_np))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_ellipsis():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is ellipsis.
    Expectation: Assert the result is equal the numpy result.
    """
    class Net(Cell):
        def construct(self, x):
            x = x[...]
            return x

    class NumpyNet():
        @classmethod
        def __call__(cls, x):
            x = x[...]
            return x

    net_ms = Net()
    net_np = NumpyNet()
    dynamic_input = Tensor(shape=(None,), dtype=mstype.float32)
    input_np = np.random.randn(4).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_bool():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is bool.
    Expectation: Assert the result is equal the numpy result.
    """
    class Net(Cell):
        def construct(self, x):
            x = x[True]
            return x

    class NumpyNet():
        @classmethod
        def __call__(cls, x):
            x = x[True]
            return x

    net_ms = Net()
    net_np = NumpyNet()
    dynamic_input = Tensor(shape=(None, 3), dtype=mstype.float32)
    input_np = np.random.randn(2, 3).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_none():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is None.
    Expectation: Assert the result is equal the numpy result.
    """
    class Net(Cell):
        def construct(self, x):
            x = x[None]
            return x

    class NumpyNet():
        @classmethod
        def __call__(cls, x):
            x = x[None]
            return x

    net_ms = Net()
    net_np = NumpyNet()
    dynamic_input = Tensor(shape=(None, 3), dtype=mstype.float32)
    input_np = np.random.randn(2, 3).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_tensor():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is tensor of int.
    Expectation: Assert the result is equal the numpy result.
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.index = Tensor([0, 1])

        def construct(self, x):
            index = self.index
            x = x[index]
            return x

    class NumpyNet():
        @classmethod
        def __call__(cls, x):
            x = x[[0, 1]]
            return x

    net_ms = Net()
    net_np = NumpyNet()
    dynamic_input = Tensor(shape=(None, 4), dtype=mstype.float32)
    input_np = np.random.randn(3, 4).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_tensor_001():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is dynamic shape tensor.
    Expectation: Assert the result is equal the numpy result.
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.unique = ops.Unique()
            self.index = Tensor([1, 1, 1, 2])

        def construct(self, x):
            index = self.unique(self.index)[0]
            x = x[index]
            return x

    class NumpyNet():
        @classmethod
        def __call__(cls, x):
            index = np.unique(np.array([1, 1, 1, 2]))
            x = x[index]
            return x

    net_ms = Net()
    net_np = NumpyNet()
    dynamic_input = Tensor(shape=(None, 3), dtype=mstype.float32)
    input_np = np.random.randn(3, 3).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_slice():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is slice.
    Expectation: Assert the result is equal the numpy result.
    """
    class Net(Cell):
        def construct(self, x):
            x = x[2:4]
            return x

    class NumpyNet():
        @classmethod
        def __call__(cls, x):
            x = x[2:4]
            return x

    net_ms = Net()
    net_np = NumpyNet()
    dynamic_input = Tensor(shape=(None, 4), dtype=mstype.float32)
    input_np = np.random.randn(6, 4).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_slice_001():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is slice with negative int.
    Expectation: Assert the result is equal the numpy result.
    """
    class Net(Cell):
        def construct(self, x):
            x = x[-3:-1]
            return x

    class NumpyNet():
        @classmethod
        def __call__(cls, x):
            x = x[-3:-1]
            return x

    net_ms = Net()
    net_np = NumpyNet()
    dynamic_input = Tensor(shape=(None, 4), dtype=mstype.float32)
    input_np = np.random.randn(6, 4).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_int():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is int.
    Expectation: Assert the result is equal the numpy result.
    """
    class Net(Cell):
        def construct(self, x):
            x = x[-3]
            return x

    class NumpyNet():
        @classmethod
        def __call__(cls, x):
            x = x[-3]
            return x

    net_ms = Net()
    net_np = NumpyNet()
    dynamic_input = Tensor(shape=(None, 4), dtype=mstype.float32)
    input_np = np.random.randn(3, 4).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_int_001():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is int with control flow.
    Expectation: Assert the result is equal the numpy result.
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.extra = 0

        def construct(self, x):
            index = 1 if self.extra > 1 else 2
            x = x[index]
            return x

    class NumpyNet():
        @classmethod
        def __call__(cls, x):
            x = x[2]
            return x

    net_ms = Net()
    net_np = NumpyNet()
    dynamic_input = Tensor(shape=(None, 2), dtype=mstype.float32)
    input_np = np.random.randn(3, 2).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_int_002():
    """
    Feature: Test Tensor slice for twice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is int.
    Expectation: Assert the result is equal the numpy result.
    """
    class Net(Cell):
        def construct(self, x):
            x = x[3][4]
            return x

    class NumpyNet():
        @classmethod
        def __call__(cls, x):
            x = x[3][4]
            return x

    net_ms = Net()
    net_np = NumpyNet()
    dynamic_input = Tensor(shape=(None, None, 3), dtype=mstype.float32)
    input_np = np.random.randn(5, 5, 3).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_list():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is list of bool and int.
    Expectation: Assert the result is equal the numpy result.
    """
    class Net(Cell):
        def construct(self, x):
            index = [False, 1]
            x = x[index]
            return x

    class NumpyNet():
        @classmethod
        def __call__(cls, x):
            index = [False, 1]
            x = x[index]
            return x

    net_ms = Net()
    net_np = NumpyNet()
    dynamic_input = Tensor(shape=(None,), dtype=mstype.float32)
    input_np = np.random.randn(5).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_tuple():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is tuple of tensor and slice.
    Expectation: Assert the result is equal the numpy result.
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.extra = Tensor(0)
            self.extra2 = Tensor(2)

        def construct(self, x):
            x = x[self.extra, self.extra:self.extra2, ...]
            return x

    class NumpyNet():
        @classmethod
        def __call__(cls, x):
            x = x[0, 0:2, ...]
            return x

    net_ms = Net()
    net_np = NumpyNet()
    dynamic_input = Tensor(shape=(2, None, 3), dtype=mstype.float32)
    input_np = np.random.randn(2, 4, 3).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_tuple_001():
    """
    Feature: Test Tensor slice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is tuple of advanced indices.
    Expectation: Assert the result is equal the numpy result.
    """
    class Net(Cell):
        def construct(self, x):
            index = (..., True, 4, slice(0, 2), None)
            x = x[index]
            return x

    class NumpyNet():
        @classmethod
        def __call__(cls, x):
            index = (..., True, 4, slice(0, 2), None)
            x = x[index]
            return x

    net_ms = Net()
    net_np = NumpyNet()
    dynamic_input = Tensor(shape=(3, 4, None, 2), dtype=mstype.float32)
    input_np = np.random.randn(3, 4, 5, 2).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_tuple_002():
    """
    Feature: Test Tensor slice for twice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is tuple of advanced indices.
    Expectation: Assert the result is equal the numpy result.
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.extra = Tensor([2, 3])

        def construct(self, x):

            x = x[True, [1, 2]][..., self.extra]
            return x

    class NumpyNet():
        @classmethod
        def __call__(cls, x):
            x = x[True, [1, 2]][..., [2, 3]]
            return x

    net_ms = Net()
    net_np = NumpyNet()
    dynamic_input = Tensor(shape=(None, 4, 5, 2, None),
                           dtype=mstype.float32)  # (1,2,4,5,2,None)
    input_np = np.random.randn(3, 4, 5, 2, 4).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_getitem_tuple_003():
    """
    Feature: Test Tensor slice for twice for dynamic shape in feed mode.
    Description: The input shape is dynamic and the tensor index is tuple of advanced indices.
    Expectation: Assert the result is equal the numpy result.
    """
    class Net(Cell):
        def construct(self, x):
            x = x[:, :, :, :1]
            return x

    class NumpyNet():
        @classmethod
        def __call__(cls, x):
            x = x[:, :, :, :1]
            return x

    net_ms = Net()
    net_np = NumpyNet()
    dynamic_input = Tensor(shape=(4, None, 5, None, 6, None),
                           dtype=mstype.float32)  # (1,2,4,5,2,None)
    input_np = np.random.randn(4, 4, 5, 5, 6, 4).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()
    context.set_context(mode=context.GRAPH_MODE)
    fact = CommonFunc(net_ms, net_np, input_np, dynamic_input)
    fact.forward_cmp()
    fact.grad_impl()
