# Copyright 2025 Huawei Technologies Co., Ltd
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
""" test_sigmoid_.py """
import numpy as np
import pytest
import torch

import mindspore as ms
from mindspore import context, Tensor
from mindspore.common import dtype as mstype
from mindspore.graph.api import _pynative_executor
from mindspore.nn import Cell

from tests.mark_utils import arg_mark
from tests.st.pynative.utils import GradOfAllInputs, allclose_nparray
from tests.st.ops.test_tools.test_op import TEST_OP


class SigmoidNet(Cell):
    def construct(self, x):
        x_clone = x.clone()
        x_clone.sigmoid_()
        return x_clone


class TorchSigmoidNet(torch.nn.Module):
    def forward(self, x):
        x_clone = x.clone()
        x_clone.sigmoid_()
        return x_clone


class TestModule():
    def __init__(self, inputs=None):
        self.ms_dtype = inputs[0].dtype
        self.input_x = inputs[0]
        self.input_x_np = inputs[0].asnumpy()

        if self.ms_dtype == mstype.float16:
            self.loss = 1e-3
        elif self.ms_dtype in (mstype.float32, mstype.complex64):
            self.loss = 1e-4
        elif self.ms_dtype in (mstype.float64, mstype.complex128):
            self.loss = 1e-5
        elif self.ms_dtype == mstype.bfloat16:
            self.loss = 4e-3
        else:
            self.loss = 0
        self.out_grad_np = None


    def forward_mindspore_impl(self):
        net = SigmoidNet()
        out = net(self.input_x)
        if out.dtype == mstype.bfloat16:
            return out.float().asnumpy()
        return out.asnumpy()

    def grad_mindspore_impl(self):
        if self.out_grad_np is None:
            out = self.forward_mindspore_impl()
            sens = np.random.randn(*list(out.shape))
            if isinstance(sens, float):
                self.out_grad_np = sens
            else:
                self.out_grad_np = sens.astype(dtype=out.dtype)
        if self.ms_dtype == mstype.bfloat16:
            ms_output_grad = Tensor(self.out_grad_np, mstype.bfloat16)
        else:
            ms_output_grad = Tensor(self.out_grad_np)
        net = SigmoidNet()
        grad_net = GradOfAllInputs(net)
        grad_net.set_train()
        input_x_grad = grad_net(self.input_x, ms_output_grad)[0]
        if input_x_grad.dtype == mstype.bfloat16:
            return input_x_grad.float().asnumpy()
        return input_x_grad.asnumpy()

    def forward_torch_impl(self):
        if self.ms_dtype == mstype.bfloat16:
            input_x = torch.from_numpy(self.input_x_np.astype(np.float32)).bfloat16()
        else:
            input_x = torch.from_numpy(self.input_x_np)

        net = TorchSigmoidNet()
        out = net(input_x)
        if self.ms_dtype == mstype.bfloat16:
            return out.detach().float().numpy()
        return out.detach().numpy()

    def grad_torch_impl(self):
        if self.ms_dtype == mstype.bfloat16:
            input_x = torch.from_numpy(self.input_x_np.astype(np.float32)).bfloat16()
        else:
            input_x = torch.from_numpy(self.input_x_np)
        input_x.requires_grad = True
        net = TorchSigmoidNet()
        out = net(input_x)
        output_grad = torch.tensor(self.out_grad_np, dtype=out.dtype)

        out.backward(output_grad)
        if self.ms_dtype == mstype.bfloat16:
            return input_x.grad.detach().float().numpy()
        return input_x.grad.detach().numpy()

    def forward_cmp(self):
        out_me = self.forward_mindspore_impl()
        out_torch = self.forward_torch_impl()
        allclose_nparray(out_torch, out_me, self.loss, self.loss)

    def grad_cmp(self):
        out_me = self.grad_mindspore_impl()
        out_torch = self.grad_torch_impl()
        allclose_nparray(out_torch, out_me, self.loss, self.loss)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('data_type', [mstype.float16, mstype.float32, mstype.float64,
                                       mstype.complex64, mstype.complex128])
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sigmoid__different_type(data_type, context_mode):
    """
    Feature: Tensor.sigmoid_ operators.
    Description: test cases for sigmoid_ operator with different data types
    Expectation: the result match between MindSpore and PyTorch.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    # Generate random test data
    shape = (3, 4)
    if data_type in (mstype.complex64, mstype.complex128):
        real_part = np.random.randn(*shape)
        imag_part = np.random.randn(*shape)
        np_data = real_part + 1j * imag_part
    else:
        np_data = np.random.randn(*shape)

    input_x = Tensor(np_data, data_type)
    module = TestModule(inputs=[input_x])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sigmoid__bf16(context_mode):
    """
    Feature: Tensor.sigmoid_ operators.
    Description: test cases for sigmoid_ operator with bfloat16 data type.
    Expectation: the result match between MindSpore and PyTorch.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.random.randn(3, 4), mstype.bfloat16)
    module = TestModule(inputs=[input_x])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('shape', [
    (3,),
    (2, 3),
    (3, 4, 5),
    (2, 3, 4, 5),
    (2, 2, 3, 4, 5),
    (2, 2, 2, 2, 2, 2),
    (2, 2, 2, 2, 2, 2, 2),
    (2, 2, 2, 2, 2, 2, 2, 2),
])
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sigmoid__different_dimensions(shape, context_mode):
    """
    Feature: Tensor.sigmoid_ operators.
    Description: test cases for sigmoid_ operator with different dimensions (1D-8D)
    Expectation: the result match between MindSpore and PyTorch.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.random.randn(*shape), mstype.float32)
    module = TestModule(inputs=[input_x])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sigmoid__empty_tensor(context_mode):
    """
    Feature: Tensor.sigmoid_ operators.
    Description: test cases for sigmoid_ operator with empty tensor
    Expectation: the result match between MindSpore and PyTorch.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.array([]).reshape(0, 3), mstype.float32)
    module = TestModule(inputs=[input_x])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sigmoid__inf_nan(context_mode):
    """
    Feature: Tensor.sigmoid_ operators.
    Description: test cases for sigmoid_ operator with inf and nan values
    Expectation: the result match between MindSpore and PyTorch.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.array([[np.inf, np.nan, 3.6], [0.4, -np.inf, -3.2]]), mstype.float32)
    module = TestModule(inputs=[input_x])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sigmoid__large_values(context_mode):
    """
    Feature: Tensor.sigmoid_ operators.
    Description: test cases for sigmoid_ operator with large values
    Expectation: the result match between MindSpore and PyTorch.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.array([[1e10, -1e10, 0], [1e-10, -1e-10, 1e5]]), mstype.float32)
    module = TestModule(inputs=[input_x])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sigmoid__single_element(context_mode):
    """
    Feature: Tensor.sigmoid_ operators.
    Description: test cases for sigmoid_ operator with single element tensor
    Expectation: the result match between MindSpore and PyTorch.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.array([[5.0]]), mstype.float32)
    module = TestModule(inputs=[input_x])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sigmoid__scalar_tensor(context_mode):
    """
    Feature: Tensor.sigmoid_ operators.
    Description: test cases for sigmoid_ operator with scalar tensor (0-dimensional)
    Expectation: the result match between MindSpore and PyTorch.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.array(5.0), mstype.float32)
    module = TestModule(inputs=[input_x])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sigmoid__zero_values(context_mode):
    """
    Feature: Tensor.sigmoid_ operators.
    Description: test cases for sigmoid_ operator with zero values
    Expectation: the result match between MindSpore and PyTorch.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mstype.float32)
    module = TestModule(inputs=[input_x])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sigmoid__mathematical_properties(context_mode):
    """
    Feature: Tensor.sigmoid_ operators.
    Description: test cases for sigmoid_ operator mathematical properties
    Expectation: the result match between MindSpore and PyTorch.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.random.randn(10, 4).astype(np.float32), mstype.float32)
    module = TestModule(inputs=[input_x])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sigmoid__input_value_range_float(context_mode):
    """
    Feature: Tensor.sigmoid_ operators.
    Description: test cases for sigmoid_ operator with random float value ranges.
    Expectation: MindSpore and PyTorch forward results are consistent; float types also verify gradients.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    # Generate random input with various value ranges
    input_x = Tensor(np.random.randn(10, 4).astype(np.float32), mstype.float32)
    module = TestModule(inputs=[input_x])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sigmoid__special_values(context_mode):
    """
    Feature: Tensor.sigmoid_ operators.
    Description: test cases for sigmoid_ operator with special values (0, large positive/negative)
    Expectation: MindSpore and PyTorch results are consistent.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.array([0.0, -10.0, 10.0, -100.0, 100.0], dtype=np.float32), mstype.float32)
    module = TestModule(inputs=[input_x])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sigmoid__input_not_tensor_raises(context_mode):
    """
    Feature: Tensor.sigmoid_ operators.
    Description: test cases for sigmoid_ operator with input_x not being Tensor type should raise exception.
    Expectation: MindSpore raises exception.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x_np = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(AttributeError):
        input_x_np.sigmoid_()
        _pynative_executor.sync()

    input_x_int = Tensor(np.array([1, 2, 3]), mstype.int32)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        input_x_int.sigmoid_()
        _pynative_executor.sync()


def sigmoid_forward_func(x):
    """Forward function for sigmoid_ operator"""
    x.clone().sigmoid_()
    return x


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
def test_sigmoid__dynamic_shape():
    """
    Feature: Tensor.sigmoid_ operators.
    Description: test cases for sigmoid_ operator using TEST_OP
    Expectation: the result match between MindSpore and PyTorch.
    """
    tensor_x1 = Tensor(np.random.randn(2, 3).astype(np.float32), mstype.float32)
    tensor_x2 = Tensor(np.random.randn(3, 4, 5).astype(np.float32), mstype.float32)

    TEST_OP(sigmoid_forward_func,
            [[tensor_x1], [tensor_x2]],
            disable_mode=['GRAPH_MODE_GE'],
            inplace_update=True)
