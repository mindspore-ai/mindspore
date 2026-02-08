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
"""Tests for mint.nn.functional.adaptive_max_pool2d."""
import numpy as np
import pytest
import torch

import mindspore as ms
from mindspore import mutable, ops, mint
from mindspore.common import Tensor
from mindspore.common import dtype as mstype
from mindspore.graph.api import _pynative_executor
from mindspore.common.dtype import _dtype_to_nptype
from mindspore.nn import Cell

from tests.mark_utils import arg_mark
from tests.st.pynative.test_pynative_embeddinglookup import OpsFactory
from tests.st.pynative.utils import allclose_nparray, GradOfFirstInput
from tests.st.utils import test_utils
from tests.st.utils.test_utils import single_golden_compare
from tests.st.ops.test_tools.test_op import TEST_OP


def _ms_forward_backward(input_x, output_size, return_indices):
    net = AdaptiveMaxPool2d()
    out = net(input_x, output_size, return_indices)
    if return_indices:
        grad_out = (ops.ones_like(out[0]), ops.zeros_like(out[1]))
    else:
        grad_out = ops.ones_like(out)
    grad_net = GradOfFirstInput(net)
    grad_net.set_train()
    grad = grad_net(input_x, output_size, return_indices, grad_out)
    return out, grad


def _torch_cpu_forward_backward(input_x_np, output_size, return_indices):
    input_x = torch.from_numpy(input_x_np).to(torch.float32)
    input_x.requires_grad = True
    net = AdaptiveMaxPool2dModule()
    out = net(input_x, output_size, return_indices)
    if return_indices:
        y, idx = out
        grad_y = (torch.ones_like(y), torch.zeros_like(idx))
        y.backward(grad_y)
        return ((y.detach().cpu(), idx.detach().cpu()), input_x.grad.detach().cpu())
    y = out
    y.backward(torch.ones_like(y))
    return (y.detach().cpu(), input_x.grad.detach().cpu())


class AdaptiveMaxPool2d(Cell):
    def construct(self, input_x, output_size, return_indices):
        out = mint.nn.functional.adaptive_max_pool2d(input_x, output_size, return_indices)
        return out


class AdaptiveMaxPool2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.torch_nn_functional_adaptive_max_pool2d = torch.nn.functional.adaptive_max_pool2d

    def forward(self, input_x, output_size, return_indices):
        out = self.torch_nn_functional_adaptive_max_pool2d(input_x, output_size, return_indices)
        return out


class AdaptiveMaxPool2dMock(OpsFactory):
    def __init__(self, attributes=None, inputs=None, grads=None):
        self.ms_type = inputs[0].dtype
        super().__init__(dtype=_dtype_to_nptype(self.ms_type))

        self.output_size = attributes.get('output_size')
        self.return_indices = attributes.get('return_indices')
        self.input_x = inputs[0]
        self.input_x_np = inputs[0].asnumpy()

        if self.ms_type == mstype.bfloat16:
            self.loss = 4e-3
        if grads is None or len(grads) == 0:
            self.out_grad_np = None
            self.out_grad_np1 = None
            self.out_grad_np2 = None
        else:
            self.out_grad_np = grads[0].asnumpy()
            self.out_grad_np1 = grads[1].asnumpy()
            self.out_grad_np2 = grads[2].asnumpy()

    def forward_mindspore_impl(self):
        net = AdaptiveMaxPool2d()
        out = net(self.input_x, self.output_size, self.return_indices)
        # If return_indices is True, returns (output, indices); otherwise only output.
        if self.return_indices and self.ms_type == mstype.bfloat16:
            return out[0].float().asnumpy(), out[1].asnumpy()
        if self.return_indices and self.ms_type != mstype.bfloat16:
            return out[0].asnumpy(), out[1].asnumpy()
        if not self.return_indices and self.ms_type == mstype.bfloat16:
            return out.float().asnumpy()
        return out.asnumpy()

    def grad_mindspore_impl(self):
        if self.out_grad_np is None:
            out = self.forward_mindspore_impl()
            if self.return_indices:
                self.out_grad_np1 = np.random.randn(*out[0].shape).astype(out[0].dtype)
                self.out_grad_np2 = np.random.randn(*out[1].shape).astype(out[1].dtype)
            else:
                sens = np.random.randn(*list(out.shape))
                self.out_grad_np = np.array(sens, dtype=out.dtype)

        if self.ms_type == mstype.bfloat16 and self.return_indices:
            out_grad_ms = (Tensor(self.out_grad_np1, mstype.bfloat16), Tensor(self.out_grad_np2))
        elif self.ms_type == mstype.bfloat16 and not self.return_indices:
            out_grad_ms = Tensor(self.out_grad_np, mstype.bfloat16)
            self.out_grad_np = out_grad_ms.float().asnumpy()
        elif self.ms_type != mstype.bfloat16 and self.return_indices:
            out_grad_ms = (Tensor(self.out_grad_np1), Tensor(self.out_grad_np2))
        else:
            out_grad_ms = Tensor(self.out_grad_np)
        net = AdaptiveMaxPool2d()
        grad_net = GradOfFirstInput(net)
        grad_net.set_train()
        grad = grad_net(self.input_x, self.output_size, self.return_indices, out_grad_ms)
        if self.ms_type == mstype.bfloat16:
            return grad.float().asnumpy()
        return grad.asnumpy()

    def forward_pytorch_impl(self):
        output_size = self.output_size
        return_indices = self.return_indices
        if self.input_x_np.dtype == np.uint16:
            input_x_np = self.input_x_np.astype(np.int16)
        elif self.input_x_np.dtype == np.uint32:
            input_x_np = self.input_x_np.astype(np.int32)
        elif self.input_x_np.dtype == np.uint64:
            input_x_np = self.input_x_np.astype(np.int64)
        elif self.input_x_np.dtype in (
                np.complex64, np.complex128, np.float64, np.int8, np.int16, np.int32,
                np.int64, np.uint8, np.bool_):
            input_x_np = self.input_x_np
        else:
            input_x_np = self.input_x_np.astype(np.float32)
        input_x = torch.from_numpy(input_x_np)
        if self.ms_type == mstype.bfloat16:
            input_x = input_x.type(torch.bfloat16)
        torch_net = AdaptiveMaxPool2dModule()

        out = torch_net(input_x, output_size, return_indices)
        # If return_indices is True, returns (output, indices); otherwise only output.
        if self.return_indices and self.ms_type == mstype.bfloat16:
            return out[0].detach().float().numpy(), out[1].detach().numpy()
        if self.return_indices and self.ms_type != mstype.bfloat16:
            return out[0].detach().numpy().astype(self.dtype), out[1].detach().numpy()
        if not self.return_indices and self.ms_type == mstype.bfloat16:
            return out.detach().float().numpy()
        return out.detach().numpy().astype(self.dtype)

    def grad_pytorch_impl(self):
        output_size = self.output_size
        return_indices = self.return_indices
        if self.input_x_np.dtype in (
                np.complex64, np.complex128, np.float64, np.int8, np.int16, np.int32,
                np.int64, np.uint8, np.bool_):
            input_x_np = self.input_x_np
        else:
            input_x_np = self.input_x_np.astype(np.float32)
        input_x = torch.from_numpy(input_x_np)
        if self.ms_type == mstype.bfloat16:
            input_x = input_x.type(torch.bfloat16)
        input_x.requires_grad = True
        torch_net = AdaptiveMaxPool2dModule()

        out = torch_net(input_x, output_size, return_indices)
        if self.return_indices:
            output_grad = (torch.from_numpy(self.out_grad_np1.astype(self.out_grad_np1.dtype)),
                           torch.from_numpy(self.out_grad_np2.astype(self.out_grad_np2.dtype)))
            out[0].backward(output_grad)
        else:
            output_grad = torch.tensor(self.out_grad_np, dtype=out.dtype)
            out.backward(output_grad)
        if self.ms_type == mstype.bfloat16:
            input_x_grad = input_x.grad.detach().float().numpy()
        else:
            input_x_grad = input_x.grad.detach().numpy().astype(self.dtype)
        return input_x_grad

    def forward_cmp(self):
        out_mindspore = self.forward_mindspore_impl()
        out_torch = self.forward_pytorch_impl()
        allclose_nparray(out_torch[0], out_mindspore[0], self.loss, self.loss)
        if self.return_indices:
            allclose_nparray(out_torch[1], out_mindspore[1], self.loss, self.loss)

    def grad_cmp(self):
        grad_mindspore = self.grad_mindspore_impl()
        grad_pytorch = self.grad_pytorch_impl()
        allclose_nparray(grad_pytorch, grad_mindspore, self.loss, self.loss)

    def forward_mindspore_dynamic_shape_impl(self, attributes, inputs):
        net = AdaptiveMaxPool2d()
        net.set_inputs(*self.dyn_inputs)
        outs = []
        for a, i in zip(attributes, inputs):
            self.__init__(attributes=a, inputs=i)
            outi = net(self.input_x, self.output_size, self.return_indices)
            outs.append(outi.asnumpy())
        return outs

    def grad_mindspore_dynamic_shape_impl(self, attributes, inputs):
        net = AdaptiveMaxPool2d()
        grad_net = GradOfFirstInput(net, sens_param=False)
        grad_net.set_inputs(*self.dyn_inputs)
        grad_net.set_train()
        grads = []
        for a, i in zip(attributes, inputs):
            self.__init__(attributes=a, inputs=i)
            grad = grad_net(self.input_x, self.output_size, self.return_indices)
            grads.append(grad.asnumpy())
        return grads

    def forward_pytorch_dynamic_shape_impl(self, attributes, inputs):
        outs = []
        for a, i in zip(attributes, inputs):
            self.__init__(attributes=a, inputs=i)
            output_size = self.output_size
            return_indices = self.return_indices
            if self.input_x_np.dtype == np.uint16:
                input_x_np = self.input_x_np.astype(np.int16)
            elif self.input_x_np.dtype == np.uint32:
                input_x_np = self.input_x_np.astype(np.int32)
            elif self.input_x_np.dtype == np.uint64:
                input_x_np = self.input_x_np.astype(np.int64)
            elif self.input_x_np.dtype in (np.bool_, np.int8, np.int16, np.int32,
                                           np.int64, np.uint8, np.uint16, np.uint32,
                                           np.uint64, np.float32, np.float64,
                                           np.complex64, np.complex128):
                input_x_np = self.input_x_np
            else:
                input_x_np = self.input_x_np.astype(np.float32)
            input_x = torch.from_numpy(input_x_np)

            torch_net = AdaptiveMaxPool2dModule()
            out = torch_net(input_x, output_size, return_indices)
            outi = out.detach().numpy().astype(self.dtype)
            outs.append(outi)
        return outs

    def grad_pytorch_dynamic_shape_impl(self, attributes, inputs):
        grads = []
        for a, i in zip(attributes, inputs):
            self.__init__(attributes=a, inputs=i)
            output_size = self.output_size
            return_indices = self.return_indices
            if self.input_x_np.dtype in (np.bool_, np.int8, np.int16, np.int32, np.int64,
                                         np.uint8, np.uint16, np.uint32, np.uint64,
                                         np.float32, np.float64, np.complex64,
                                         np.complex128):
                input_x_np = self.input_x_np
            else:
                input_x_np = self.input_x_np.astype(np.float32)
            input_x = torch.from_numpy(input_x_np)
            input_x.requires_grad = True

            torch_net = AdaptiveMaxPool2dModule()
            out = torch_net(input_x, output_size, return_indices)
            output_grad = torch.ones_like(out)
            out.backward(output_grad)
            input_x_grad = input_x.grad.detach().numpy()
            gradi = input_x_grad.astype(self.dtype)
            grads.append(gradi)
        return grads

    def forward_dynamic_shape_cmp(self, attributes, inputs):
        out_ms = self.forward_mindspore_dynamic_shape_impl(attributes, inputs)
        out_cmp = self.forward_pytorch_dynamic_shape_impl(attributes, inputs)
        for a, b in zip(out_cmp, out_ms):
            allclose_nparray(a, b, self.loss, self.loss)

    def grad_dynamic_shape_cmp(self, attributes, inputs):
        grad_ms = self.grad_mindspore_dynamic_shape_impl(attributes, inputs)
        grad_cmp = self.grad_pytorch_dynamic_shape_impl(attributes, inputs)
        for a, b in zip(grad_cmp, grad_ms):
            allclose_nparray(a, b, self.loss, self.loss)


def set_context_mode(mode):
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'kbk':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O2')


@test_utils.run_with_cell
def adaptive_max_pool2d_forward_dyn_func(input_x):
    return mint.nn.functional.adaptive_max_pool2d(input_x, (8, 8), False)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=20.01)
@pytest.mark.parametrize('return_indices', [True, False])
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_bfloat16_3d_9x4x9_random(mode, return_indices):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: 3D bfloat16 input, output_size=(21, 50); compare with PyTorch.
    Expectation: forward and gradient match PyTorch within tolerance.
    """
    set_context_mode(mode)
    input_x_np = np.random.randn(9, 4, 9).astype(np.float32)
    output_size = (21, 50)
    input_x_np = torch.from_numpy(input_x_np).to(torch.bfloat16).to(torch.float32).numpy()
    ms_input_x = Tensor(input_x_np, dtype=mstype.bfloat16)
    ms_y, ms_gx = _ms_forward_backward(ms_input_x, output_size, return_indices)
    th_y_fp32, th_gx_fp32 = _torch_cpu_forward_backward(input_x_np, output_size, return_indices)
    if return_indices:
        assert single_golden_compare(th_y_fp32[0], ms_y[0], 3825)
        # The second tensor is indices; do not use single golden compare.
        # Use allclose_nparray for comparison.
        allclose_nparray(th_y_fp32[1].detach().numpy(), ms_y[1].asnumpy(), 0, 0)
        assert single_golden_compare(th_gx_fp32, ms_gx, 9450)
        return
    assert single_golden_compare(th_y_fp32, ms_y, 3825)
    assert single_golden_compare(th_gx_fp32, ms_gx, 9450)


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=20.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_float32_3d_9x4x9_random(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: test adaptive_max_pool2d with float32 input and output_size (21, 50) and return_indices False.
    Expectation: expect correct result.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randn(9, 4, 9), mstype.float32)
    output_size = (21, 50)
    return_indices = False
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=20.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_float32_3d_9x4x9_random_none_none(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: 3D float32 input, output_size=(None, None), return_indices=False; compare with PyTorch.
    Expectation: forward and gradient match PyTorch within tolerance.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randn(9, 4, 9), mstype.float32)
    output_size = (None, None)
    return_indices = False
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=20.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_float32_3d_9x4x9_random_none(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: 3D float32 input, output_size=(21, None), return_indices=False; compare with PyTorch.
    Expectation: forward and gradient match PyTorch within tolerance.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randn(9, 4, 9), mstype.float32)
    output_size = (21, None)
    return_indices = False
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=20.0)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_float16_4d_4x7x7x5_random(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: 4D float16 input, output_size=5, return_indices=False; compare with PyTorch.
    Expectation: forward and gradient match PyTorch within relaxed tolerance.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randn(4, 7, 7, 5), mstype.float16)
    output_size = 5
    return_indices = False
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    fact.forward_cmp()
    # The operator accumulates values; for large shapes, fp16 vs fp32 accumulation error may be larger.
    fact.loss = 0.01
    fact.grad_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_float64_4d_8x5x6x9_random(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: 4D float64 input, output_size=(5, 4), return_indices=False; compare with PyTorch.
    Expectation: forward and gradient match PyTorch within tolerance.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randn(8, 5, 6, 9), mstype.float64)
    output_size = (5, 4)
    return_indices = False
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    fact.forward_cmp()
    #Ascend not support float64 backward, so only test forward.
    if ms.context.get_context("device_target") != "Ascend":
        fact.grad_cmp()


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_float32_4d_3x3x8x3_random(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: 4D float32 input, output_size=46, return_indices=False; compare with PyTorch.
    Expectation: forward and gradient match PyTorch within tolerance.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randn(3, 3, 8, 3), mstype.float32)
    output_size = 46
    return_indices = False
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_float16_3d_7x9x3_random(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: 3D float16 input, output_size=(21, 5), return_indices=False; compare with PyTorch.
    Expectation: forward and gradient match PyTorch within relaxed tolerance.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randn(7, 9, 3), mstype.float16)
    output_size = (21, 5)
    return_indices = False
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    fact.forward_cmp()
    # The operator accumulates values; for large shapes, fp16 vs fp32 accumulation error may be larger.
    fact.loss = 0.01
    fact.grad_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_float64_3d_5x8x4_random(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: 3D float64 input, output_size=50, return_indices=False; compare with PyTorch.
    Expectation: forward and gradient match PyTorch within tolerance.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randn(5, 8, 4), mstype.float64)
    output_size = 50
    return_indices = False
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    fact.forward_cmp()
    #Ascend not support float64 backward, so only test forward.
    if ms.context.get_context("device_target") != "Ascend":
        fact.grad_cmp()



@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_float32_3d_1x4x9_random(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: 3D float32 input (N=1), output_size=(5, 6), return_indices=False; compare with PyTorch.
    Expectation: forward and gradient match PyTorch within tolerance.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randn(1, 4, 9), mstype.float32)
    output_size = (5, 6)
    return_indices = False
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_float16_4d_2x6x6x1_return_indices_true(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: 4D float16 input, output_size=3, return_indices=True; compare values and indices with PyTorch.
    Expectation: forward outputs and indices, and gradient match within relaxed tolerance.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randn(2, 6, 6, 1).astype(np.float16))
    output_size = 3
    return_indices = True
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    fact.forward_cmp()
    # The operator accumulates values; for large shapes, fp16 vs fp32 accumulation error may be larger.
    fact.loss = 0.01
    fact.grad_cmp()


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_float32_4d_1x1x3x3_return_indices_true(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: 4D float32 input (1x1x3x3), output_size=(1, 3), return_indices=True; compare with PyTorch.
    Expectation: forward outputs and indices, and gradient match PyTorch.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randn(1, 1, 3, 3), mstype.float32)
    output_size = (1, 3)
    return_indices = True
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_float16_4d_2x7x1x6_random(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: 4D float16 input, output_size=4, return_indices=False; compare with PyTorch.
    Expectation: forward and gradient match PyTorch within relaxed tolerance.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randn(2, 7, 1, 6), mstype.float16)
    output_size = 4
    return_indices = False
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    fact.forward_cmp()
    # The operator accumulates values; for large shapes, fp16 vs fp32 accumulation error may be larger.
    fact.loss = 0.01
    fact.grad_cmp()


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level3',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_input_not_tensor(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: input is not Tensor (float), expect type check.
    Expectation: raise TypeError.
    """
    set_context_mode(mode)
    input_x = 1.0
    output_size = (21, 50)
    return_indices = False
    net = AdaptiveMaxPool2d()
    with pytest.raises(TypeError):
        net(input_x, output_size, return_indices)


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level3',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_output_size_float(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: output_size is float, expect type check.
    Expectation: raise TypeError.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randn(9, 4, 9), mstype.float32)
    output_size = 1.0
    return_indices = False
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    with pytest.raises(TypeError):
        fact.forward_mindspore_impl()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level3',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_return_indices_float(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: return_indices is float, expect type check during execution.
    Expectation: raise TypeError.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randn(9, 4, 9), mstype.float32)
    output_size = (21, 50)
    return_indices = 1.0
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    with pytest.raises(TypeError):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level3',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_input_dtype_uint64(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: input dtype uint64 is unsupported.
    Expectation: raise TypeError or RuntimeError.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randint(0, 1000, (9, 4, 9)), mstype.uint64)

    output_size = (21, 50)
    return_indices = False
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    with pytest.raises((TypeError, RuntimeError)):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01
          )
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_input_rank_1d_invalid(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: input rank=1, expect rank check failure.
    Expectation: raise RuntimeError/ValueError/AssertionError.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randn(9, ), mstype.float32)
    output_size = (21, 50)
    return_indices = False
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level3',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_input_1d_0_none_invalid(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: empty tensor case (shape[1]==0) with 1D-like input helper.
    Expectation: raise RuntimeError/ValueError/AssertionError.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randn(*(2, 0)), mstype.float32)
    output_size = (21, 50)
    return_indices = False
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level3',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_output_size_length_not2(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: output_size tuple length is 3, expect validation failure.
    Expectation: raise RuntimeError/ValueError/AssertionError.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randn(9, 4, 9), mstype.float32)
    output_size = (21, 50, 50)
    return_indices = False
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level3',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_output_size_element_out_range(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: output_size contains -2, which is out of allowed range.
    Expectation: raise RuntimeError/ValueError/AssertionError.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randn(9, 4, 9), mstype.float32)
    output_size = (-2, 50)
    return_indices = False
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level3',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_output_size_out_range(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: output_size = -2 (computed), which is invalid.
    Expectation: raise RuntimeError/ValueError/AssertionError.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randn(9, 4, 9), mstype.float32)
    output_size = -1 - 1
    return_indices = False
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_float32_4d_8x4x9x9_nan(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: 4D float32 input filled with NaN; run forward/backward only.
    Expectation: complete execution without crash.
    """
    set_context_mode(mode)
    input_x = Tensor(np.full((8, 4, 9, 9), np.nan), mstype.float32)
    output_size = 26
    return_indices = False
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    fact.forward_mindspore_impl()
    fact.grad_mindspore_impl()


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_float32_3d_9x3x5_inf(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: 3D float32 input filled with Inf; compare with PyTorch.
    Expectation: forward and gradient match PyTorch.
    """
    set_context_mode(mode)
    input_x = Tensor(np.full((9, 3, 5), np.inf), mstype.float32)
    output_size = (50, 32)
    return_indices = False
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_dynamic_shape_f_adaptive_max_pool2d_dyn_shape_1(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d (dynamic shape)
    Description: 4D float16 dynamic shape; run three variants with mutable output_size=int.
    Expectation: forward/grad run correctly for all variants.
    """
    set_context_mode(mode)
    input_x = Tensor(shape=(None, None, None, None), dtype=mstype.float16)
    output_size = mutable(input_data=6, dynamic_len=False)
    return_indices = False  # In KBK mode, do not support mutable(return_indices)
    input_x1 = Tensor(np.random.randn(9, 4, 7, 9), mstype.float16)
    output_size1 = mutable(input_data=6, dynamic_len=False)
    return_indices1 = False
    attributes1 = {'output_size': output_size1, 'return_indices': return_indices1}
    inputs1 = [input_x1]
    input_x2 = Tensor(np.random.randn(8, 8, 3, 3), mstype.float16)
    output_size2 = mutable(input_data=9, dynamic_len=False)
    return_indices2 = False
    attributes2 = {'output_size': output_size2, 'return_indices': return_indices2}
    inputs2 = [input_x2]
    input_x3 = Tensor(np.random.randn(3, 4, 8, 9), mstype.float16)
    output_size3 = mutable(input_data=7, dynamic_len=False)
    return_indices3 = False
    attributes3 = {'output_size': output_size3, 'return_indices': return_indices3}
    inputs3 = [input_x3]
    all_attrs = [attributes1, attributes2, attributes3]
    all_inputs = [inputs1, inputs2, inputs3]
    fact = AdaptiveMaxPool2dMock(attributes=attributes1, inputs=inputs1)
    fact.dyn_inputs = (input_x, output_size, return_indices)
    fact.forward_dynamic_shape_cmp(all_attrs, all_inputs)
    fact.grad_dynamic_shape_cmp(all_attrs, all_inputs)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_dynamic_shape_f_adaptive_max_pool2d_dyn_shape_2(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d (dynamic shape)
    Description: 4D float16 dynamic shape; run three variants with mutable output_size=(H, W).
    Expectation: forward/grad run correctly for all variants.
    """
    set_context_mode(mode)
    input_x = Tensor(shape=(None, None, None, None), dtype=mstype.float16)
    output_size = mutable(input_data=(3, 4), dynamic_len=False)
    return_indices = False
    input_x1 = Tensor(np.random.randn(5, 4, 7, 5), mstype.float16)
    output_size1 = mutable(input_data=(3, 3), dynamic_len=False)
    return_indices1 = False
    attributes1 = {'output_size': output_size1, 'return_indices': return_indices1}
    inputs1 = [input_x1]
    input_x2 = Tensor(np.random.randn(7, 5, 8, 3), mstype.float16)
    output_size2 = mutable(input_data=(2, 1), dynamic_len=False)
    return_indices2 = False
    attributes2 = {'output_size': output_size2, 'return_indices': return_indices2}
    inputs2 = [input_x2]
    input_x3 = Tensor(np.random.randn(3, 8, 3, 3), mstype.float16)
    output_size3 = mutable(input_data=(4, 6), dynamic_len=False)
    return_indices3 = False
    attributes3 = {'output_size': output_size3, 'return_indices': return_indices3}
    inputs3 = [input_x3]
    all_attrs = [attributes1, attributes2, attributes3]
    all_inputs = [inputs1, inputs2, inputs3]
    fact = AdaptiveMaxPool2dMock(attributes=attributes1, inputs=inputs1)
    fact.dyn_inputs = (input_x, output_size, return_indices)
    fact.forward_dynamic_shape_cmp(all_attrs, all_inputs)
    fact.grad_dynamic_shape_cmp(all_attrs, all_inputs)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_dynamic_shape_f_adaptive_max_pool2d_dyn_shape_3(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d (dynamic shape)
    Description: 3D float16 dynamic shape; run three variants with mutable output_size=(H, W).
    Expectation: forward/grad run correctly for all variants.
    """
    set_context_mode(mode)
    input_x = Tensor(shape=(None, None, None), dtype=mstype.float16)
    output_size = mutable(input_data=(3, 4), dynamic_len=False)
    return_indices = False
    input_x1 = Tensor(np.random.randn(8, 8, 7), mstype.float16)
    output_size1 = mutable(input_data=(2, 4), dynamic_len=False)
    return_indices1 = False
    attributes1 = {'output_size': output_size1, 'return_indices': return_indices1}
    inputs1 = [input_x1]
    input_x2 = Tensor(np.random.randn(4, 8, 5), mstype.float16)
    output_size2 = mutable(input_data=(5, 4), dynamic_len=False)
    return_indices2 = False
    attributes2 = {'output_size': output_size2, 'return_indices': return_indices2}
    inputs2 = [input_x2]
    input_x3 = Tensor(np.random.randn(7, 9, 5), mstype.float16)
    output_size3 = mutable(input_data=(5, 2), dynamic_len=False)
    return_indices3 = False
    attributes3 = {'output_size': output_size3, 'return_indices': return_indices3}
    inputs3 = [input_x3]
    all_attrs = [attributes1, attributes2, attributes3]
    all_inputs = [inputs1, inputs2, inputs3]
    fact = AdaptiveMaxPool2dMock(attributes=attributes1, inputs=inputs1)
    fact.dyn_inputs = (input_x, output_size, return_indices)
    fact.forward_dynamic_shape_cmp(all_attrs, all_inputs)
    fact.grad_dynamic_shape_cmp(all_attrs, all_inputs)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_dynamic_shape_f_adaptive_max_pool2d_dyn_rank_1(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d (dynamic rank)
    Description: dynamic rank input Tensor(None) with output_size=int; run three variants.
    Expectation: forward/grad run correctly for all variants.
    """
    set_context_mode(mode)
    input_x = Tensor(None, dtype=mstype.float16)
    output_size = mutable(input_data=7, dynamic_len=False)
    return_indices = False
    input_x1 = Tensor(np.random.randn(5, 5, 4, 7), mstype.float16)
    output_size1 = mutable(input_data=6, dynamic_len=False)
    return_indices1 = False
    attributes1 = {'output_size': output_size1, 'return_indices': return_indices1}
    inputs1 = [input_x1]
    input_x2 = Tensor(np.random.randn(8, 8, 4), mstype.float16)
    output_size2 = mutable(input_data=9, dynamic_len=False)
    return_indices2 = False
    attributes2 = {'output_size': output_size2, 'return_indices': return_indices2}
    inputs2 = [input_x2]
    input_x3 = Tensor(np.random.randn(9, 3, 8, 7), mstype.float16)
    output_size3 = mutable(input_data=8, dynamic_len=False)
    return_indices3 = False
    attributes3 = {'output_size': output_size3, 'return_indices': return_indices3}
    inputs3 = [input_x3]
    all_attrs = [attributes1, attributes2, attributes3]
    all_inputs = [inputs1, inputs2, inputs3]
    fact = AdaptiveMaxPool2dMock(attributes=attributes1, inputs=inputs1)
    fact.dyn_inputs = (input_x, output_size, return_indices)
    fact.forward_dynamic_shape_cmp(all_attrs, all_inputs)
    fact.grad_dynamic_shape_cmp(all_attrs, all_inputs)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_dynamic_shape_f_adaptive_max_pool2d_dyn_rank_2(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d (dynamic rank)
    Description: dynamic rank input Tensor(None) with output_size=(H, W); run three variants.
    Expectation: forward/grad run correctly for all variants.
    """
    set_context_mode(mode)
    input_x = Tensor(None, dtype=mstype.float16)
    output_size = mutable(input_data=(3, 2), dynamic_len=False)
    return_indices = False
    input_x1 = Tensor(np.random.randn(5, 5, 4, 7), mstype.float16)
    output_size1 = mutable(input_data=(3, 2), dynamic_len=False)
    return_indices1 = False
    attributes1 = {'output_size': output_size1, 'return_indices': return_indices1}
    inputs1 = [input_x1]
    input_x2 = Tensor(np.random.randn(8, 8, 4), mstype.float16)
    output_size2 = mutable(input_data=(3, 3), dynamic_len=False)
    return_indices2 = False
    attributes2 = {'output_size': output_size2, 'return_indices': return_indices2}
    inputs2 = [input_x2]
    input_x3 = Tensor(np.random.randn(9, 3, 8, 7), mstype.float16)
    output_size3 = mutable(input_data=(3, 4), dynamic_len=False)
    return_indices3 = False
    attributes3 = {'output_size': output_size3, 'return_indices': return_indices3}
    inputs3 = [input_x3]
    all_attrs = [attributes1, attributes2, attributes3]
    all_inputs = [inputs1, inputs2, inputs3]
    fact = AdaptiveMaxPool2dMock(attributes=attributes1, inputs=inputs1)
    fact.dyn_inputs = (input_x, output_size, return_indices)
    fact.forward_dynamic_shape_cmp(all_attrs, all_inputs)
    fact.grad_dynamic_shape_cmp(all_attrs, all_inputs)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=20.0)
def test_ops_adaptive_max_pool2d_dynamic_shape():
    """
    Feature: mint.nn.functional.adaptive_max_pool2d (dynamic shape)
    Description: validate TEST_OP pipeline for dynamic shape with two input cases.
    Expectation: both cases run successfully; shape/dtype checks pass.
    """
    input_np_1 = np.random.randn(*(6, 4, 8, 9)).astype(np.float32)
    input_np_2 = np.random.randn(*(3, 7, 8, 5)).astype(np.float32)
    TEST_OP(adaptive_max_pool2d_forward_dyn_func, [[Tensor(input_np_1)], [Tensor(input_np_2)]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True})

@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_f_adaptive_max_pool2d_float32_empty_tensor(mode):
    """
    Feature: mint.nn.functional.adaptive_max_pool2d
    Description: test adaptive_max_pool2d with empty input.
    Expectation: expect correct result.
    """
    set_context_mode(mode)
    input_x = Tensor(np.random.randn(0, 1, 4, 5), mstype.float32)
    output_size = (8, 8)
    return_indices = False
    fact = AdaptiveMaxPool2dMock(
        attributes={'output_size': output_size, 'return_indices': return_indices},
        inputs=[input_x])
    ms_out_np = fact.forward_mindspore_impl()
    th_out_np = fact.forward_pytorch_impl()
    assert np.array_equal(ms_out_np, th_out_np)
    fact.grad_cmp()
