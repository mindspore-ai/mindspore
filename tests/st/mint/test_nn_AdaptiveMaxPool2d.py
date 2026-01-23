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
"""Tests for nn.AdaptiveMaxPool2d."""
import numpy as np
import pytest
import torch
from torch import nn

import mindspore as ms
from mindspore.common import Tensor
from mindspore.graph.api import _pynative_executor
from mindspore.nn import AdaptiveMaxPool2d

from tests.mark_utils import arg_mark
from tests.st.pynative.test_pynative_embeddinglookup import OpsFactory
from tests.st.pynative.utils import allclose_nparray, GradOfFirstInput

class AdaptiveMaxPool2dFactory(OpsFactory):
    def __init__(self, input_shape, output_size, return_indices=True, dtype=np.float16):
        super().__init__(dtype=dtype)
        self.input_np = np.random.randn(*input_shape).astype(dtype)
        self.output_size = output_size
        self.return_indices = return_indices
        self.dtype = dtype
        self.output_grad_np = None

    def forward_mindspore_impl(self):
        input_x = Tensor(self.input_np.copy())
        net = AdaptiveMaxPool2d(output_size=self.output_size, return_indices=self.return_indices)
        if self.return_indices:
            output, max_axis = net(input_x)
            return output.asnumpy(), max_axis.asnumpy().astype(np.int32)
        output = net(input_x)
        return output.asnumpy()

    def forward_pytorch_impl(self):
        input_x = torch.from_numpy(self.input_np.copy().astype(np.float32))
        net = nn.AdaptiveMaxPool2d(self.output_size, self.return_indices)
        if self.return_indices:
            output, max_axis = net(input_x)
            return output.detach().numpy().astype(self.dtype), \
                   max_axis.detach().numpy().astype(np.int32)
        output = net(input_x)
        return output.detach().numpy().astype(self.dtype)

    def forward_cmp(self, skip_argmax=False):
        out_torch = self.forward_pytorch_impl()
        out_me = self.forward_mindspore_impl()
        if isinstance(out_me, tuple):
            allclose_nparray(out_torch[0], out_me[0], self.loss, self.loss)
            if not skip_argmax:
                allclose_nparray(out_torch[1], out_me[1], self.loss, self.loss)
        else:
            allclose_nparray(out_torch, out_me, self.loss, self.loss)

    def grad_mindspore_impl(self):
        input_me = Tensor(self.input_np.copy())
        tmp_out = self.forward_mindspore_impl()
        self.output_grad_np = (tmp_out[0].astype(self.dtype), tmp_out[1].astype(np.int64))
        output_grad_1 = Tensor(self.output_grad_np[0])
        output_grad_2 = Tensor(self.output_grad_np[1])

        net = AdaptiveMaxPool2d(self.output_size, True)
        grad_net = GradOfFirstInput(net, real_inputs_count=1)
        grad_net.set_train()
        input_grad = grad_net(input_me, output_grad_1, output_grad_2)
        return input_grad.asnumpy()

    def grad_pytorch_impl(self):
        # prepare input
        input_pt = torch.from_numpy(self.input_np.copy().astype(np.float32))
        input_pt.requires_grad = True
        # # prepare out_grad index
        out_grad = torch.from_numpy(self.output_grad_np[0].copy().astype(np.float32))

        # cal grad
        net = torch.nn.AdaptiveMaxPool2d(self.output_size, True)
        out = net(input_pt)
        out[0].backward(out_grad)
        return input_pt.grad.numpy().astype(np.float32)

    def grad_cmp(self):
        input_grad_ms = self.grad_mindspore_impl()
        input_grad_pt = self.grad_pytorch_impl()
        allclose_nparray(input_grad_pt, input_grad_ms, self.loss, self.loss)


def set_context_mode(mode):
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'kbk':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O2')


@arg_mark(plat_marks=['platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=20.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_n_adaptivemaxpool2d_input_2x64x32x32_output_size_16x16_return_indices_true_fp16(mode):
    """
    Feature: nn.AdaptiveMaxPool2d
    Description: test nn.AdaptiveMaxPool2d with float16 input and output_size (16, 16) and return_indices True.
    Expectation: expect correct result.
    """
    set_context_mode(mode)
    input_shape = (2, 64, 32, 32)
    output_size = 16
    return_indices = True
    fact = AdaptiveMaxPool2dFactory(input_shape, output_size, return_indices, dtype=np.float16)
    fact.forward_cmp(skip_argmax=True)
    fact.grad_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=20.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_n_adaptivemaxpool2d_input_2x64x32x32_output_size_16x16_return_indices_false_fp16(mode):
    """
    Feature: nn.AdaptiveMaxPool2d
    Description: 4D float16 input, output_size=16, return_indices=False; compare with PyTorch.
    Expectation: forward matches PyTorch.
    """
    set_context_mode(mode)
    input_shape = (2, 64, 32, 32)
    output_size = 16
    return_indices = False
    fact = AdaptiveMaxPool2dFactory(input_shape, output_size, return_indices)
    fact.forward_cmp(skip_argmax=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_n_adaptivemaxpool2d_input_2x64x32x32_output_size_nonexnone_return_indices_false_fp16(mode):
    """
    Feature: nn.AdaptiveMaxPool2d
    Description: 4D float16 input, output_size=(None, None), return_indices=False; compare with PyTorch.
    Expectation: forward matches PyTorch.
    """
    set_context_mode(mode)
    input_shape = (2, 64, 32, 32)
    output_size = (None, None)
    return_indices = False
    fact = AdaptiveMaxPool2dFactory(input_shape, output_size, return_indices)
    fact.forward_cmp(skip_argmax=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=20.0)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_n_adaptivemaxpool2d_input_2x64x32x32_output_size_nonex10_return_indices_false_fp16(mode):
    """
    Feature: nn.AdaptiveMaxPool2d
    Description: 4D float16 input, output_size=(None, 10), return_indices=False; compare with PyTorch.
    Expectation: forward matches PyTorch.
    """
    set_context_mode(mode)
    input_shape = (2, 64, 32, 32)
    output_size = (None, 10)
    return_indices = False
    fact = AdaptiveMaxPool2dFactory(input_shape, output_size, return_indices)
    fact.forward_cmp(skip_argmax=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_n_adaptivemaxpool2d_input_2x64x32x32_output_size_10xnone_return_indices_false_fp16(mode):
    """
    Feature: nn.AdaptiveMaxPool2d
    Description: 4D float16 input, output_size=(10, None), return_indices=False; compare with PyTorch.
    Expectation: forward matches PyTorch.
    """
    set_context_mode(mode)
    input_shape = (2, 64, 32, 32)
    output_size = (10, None)
    return_indices = False
    fact = AdaptiveMaxPool2dFactory(input_shape, output_size, return_indices)
    fact.forward_cmp(skip_argmax=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_n_adaptivemaxpool2d_input_32x64x128x64_output_size_32x30_return_indices_false_fp16(mode):
    """
    Feature: nn.AdaptiveMaxPool2d
    Description: Large 4D float16 input, output_size=(32, 30), return_indices=False; compare with PyTorch.
    Expectation: forward matches PyTorch.
    """
    set_context_mode(mode)
    input_shape = (32, 64, 128, 64)
    output_size = (32, 30)
    return_indices = False
    fact = AdaptiveMaxPool2dFactory(input_shape, output_size, return_indices)
    fact.forward_cmp(skip_argmax=True)


@arg_mark(plat_marks=['platform_gpu',
                      'cpu_linux'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_n_adaptivemaxpool2d_input_32x64x128x64_output_size_32x30_return_indices_true_fp16(mode):
    """
    Feature: nn.AdaptiveMaxPool2d
    Description: Large 4D float16 input, output_size=(32, 30), return_indices=True; compare values and indices.
    Expectation: forward values/indices and gradient match PyTorch.
    """
    set_context_mode(mode)
    input_shape = (32, 64, 128, 64)
    output_size = (32, 30)
    return_indices = True
    fact = AdaptiveMaxPool2dFactory(input_shape, output_size, return_indices)
    fact.forward_cmp(skip_argmax=False)
    fact.grad_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level3',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_n_adaptivemaxpool2d_input_2x64x32x32_output_size_10x10_return_indices_1_fp16(mode):
    """
    Feature: nn.AdaptiveMaxPool2d
    Description: Invalid return_indices type (int).
    Expectation: raise TypeError.
    """
    set_context_mode(mode)
    input_shape = (2, 64, 32, 32)
    output_size = 10
    return_indices = 1
    fact = AdaptiveMaxPool2dFactory(input_shape, output_size, return_indices)
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
def test_n_adaptivemaxpool2d_input_2x64x32x32_output_size_str_return_indices_false_fp16(mode):
    """
    Feature: nn.AdaptiveMaxPool2d
    Description: Invalid output_size type (str).
    Expectation: raise TypeError.
    """
    set_context_mode(mode)
    input_shape = (2, 64, 32, 32)
    output_size = "is output"
    return_indices = True
    fact = AdaptiveMaxPool2dFactory(input_shape, output_size, return_indices)
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
def test_n_adaptivemaxpool2d_input_2x64x32x32_output_size_negtive20_return_indices_false_fp16(mode):
    """
    Feature: nn.AdaptiveMaxPool2d
    Description: Invalid output_size value (-20).
    Expectation: raise RuntimeError/ValueError.
    """
    set_context_mode(mode)
    input_shape = (2, 64, 32, 32)
    output_size = -20
    return_indices = True
    fact = AdaptiveMaxPool2dFactory(input_shape, output_size, return_indices)
    with pytest.raises((RuntimeError, ValueError)):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_n_adaptivemaxpool2d_input_32x2x256x256_output_size_64_return_indices_false_fp16(mode):
    """
    Feature: nn.AdaptiveMaxPool2d
    Description: 4D float16 input, output_size=64, return_indices=False; compare with PyTorch.
    Expectation: forward matches PyTorch.
    """
    set_context_mode(mode)
    input_shape = (32, 2, 256, 256)
    output_size = 64
    return_indices = False
    fact = AdaptiveMaxPool2dFactory(input_shape, output_size, return_indices)
    fact.forward_cmp(skip_argmax=True)


@arg_mark(plat_marks=['platform_gpu',
                      'cpu_linux'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_n_adaptivemaxpool2d_input_32x2x256x256_output_size_64_return_indices_false_fp32(mode):
    """
    Feature: nn.AdaptiveMaxPool2d
    Description: 4D float32 input, output_size=64, return_indices=False; compare with PyTorch.
    Expectation: forward matches PyTorch.
    """
    set_context_mode(mode)
    input_shape = (32, 2, 256, 256)
    output_size = 64
    return_indices = False
    fact = AdaptiveMaxPool2dFactory(input_shape, output_size, return_indices, np.float32)
    fact.forward_cmp(skip_argmax=True)


@arg_mark(plat_marks=['platform_gpu',
                      'cpu_linux'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_n_adaptivemaxpool2d_input_32x2x6x6_output_size_4_indices_false_fp32(mode):
    """
    Feature: nn.AdaptiveMaxPool2d
    Description: 4D float32 input, output_size=4, return_indices=False; compare with PyTorch.
    Expectation: forward matches PyTorch.
    """
    set_context_mode(mode)
    input_shape = (32, 2, 6, 6)
    output_size = 4
    return_indices = False
    fact = AdaptiveMaxPool2dFactory(input_shape, output_size, return_indices, np.float32)
    fact.forward_cmp(skip_argmax=True)


@arg_mark(plat_marks=['platform_gpu',
                      'cpu_linux'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_n_adaptivemaxpool2d_input_32x256x128x256_output_size_32x64_double_mode_fp32(mode):
    """
    Feature: nn.AdaptiveMaxPool2d
    Description: 4D float32 input, output_size=(32, 64), return_indices=True; forward-only compare.
    Expectation: forward matches PyTorch.
    """
    set_context_mode(mode)
    input_shape = (32, 256, 128, 256)
    output_size = (32, 64)
    return_indices = True
    fact = AdaptiveMaxPool2dFactory(input_shape, output_size, return_indices, np.float32)
    fact.forward_cmp(skip_argmax=True)


@arg_mark(plat_marks=['platform_gpu',
                      'cpu_linux'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_n_adaptivemaxpool2d_input_32x256x128x256_output_size_32x64_double_mode_fp64(mode):
    """
    Feature: nn.AdaptiveMaxPool2d
    Description: 4D float64 input, output_size=(32, 64), return_indices=True; forward-only compare.
    Expectation: forward matches PyTorch.
    """
    set_context_mode(mode)
    input_shape = (32, 256, 128, 256)
    output_size = (32, 64)
    return_indices = True
    fact = AdaptiveMaxPool2dFactory(input_shape, output_size, return_indices, np.float64)
    fact.forward_cmp(skip_argmax=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_n_adaptivemaxpool2d_input_32x2x6x6_output_size_4_return_indices_false_fp16(mode):
    """
    Feature: nn.AdaptiveMaxPool2d
    Description: 4D float16 input, output_size=4, return_indices=False; compare with PyTorch.
    Expectation: forward matches PyTorch.
    """
    set_context_mode(mode)
    input_shape = (32, 2, 6, 6)
    output_size = 4
    return_indices = False
    fact = AdaptiveMaxPool2dFactory(input_shape, output_size, return_indices)
    fact.forward_cmp(skip_argmax=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_n_adaptivemaxpool2d_input_1x1x1x1_output_size_1_return_indices_false_fp16(mode):
    """
    Feature: nn.AdaptiveMaxPool2d
    Description: Small 4D float16 input (1x1x1x1), output_size=1, return_indices=False; compare with PyTorch.
    Expectation: forward matches PyTorch.
    """
    set_context_mode(mode)
    input_shape = (1, 1, 1, 1)
    output_size = 1
    return_indices = False
    fact = AdaptiveMaxPool2dFactory(input_shape, output_size, return_indices)
    fact.forward_cmp(skip_argmax=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level3',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_n_adaptivemaxpool2d_input_1x1x1x1_output_size_1_return_indices_false_int32(mode):
    """
    Feature: nn.AdaptiveMaxPool2d
    Description: Invalid input dtype int32.
    Expectation: raise RuntimeError/TypeError.
    """
    set_context_mode(mode)
    input_shape = (1, 1, 1, 1)
    output_size = 1
    return_indices = False
    fact = AdaptiveMaxPool2dFactory(input_shape, output_size, return_indices, np.int32)
    with pytest.raises((RuntimeError, TypeError)):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux'],
          level_mark='level3',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_n_adaptivemaxpool2d_input_1x1x1x1_output_size_1_return_indices_false_bool(mode):
    """
    Feature: nn.AdaptiveMaxPool2d
    Description: Invalid input dtype bool.
    Expectation: raise RuntimeError/TypeError.
    """
    set_context_mode(mode)
    input_shape = (1, 1, 1, 1)
    output_size = 1
    return_indices = False
    fact = AdaptiveMaxPool2dFactory(input_shape, output_size, return_indices, np.bool_)
    with pytest.raises((RuntimeError, TypeError)):
        fact.forward_mindspore_impl()
        _pynative_executor.sync()
