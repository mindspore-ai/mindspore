# Copyright 2025 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import mindspore as ms
from mindspore import mint

from mindspore import Tensor, context
from tests.mark_utils import arg_mark


def _bn_fwd(x, weight, bias, mean, var, training=False,
            momentum=0.1, eps=1e-5):
    return mint.nn.functional.batch_norm(
        x, mean, var, weight, bias, training, momentum, eps
    )


def _grad_x(x, weight, bias, mean, var, training=False,
            momentum=0.1, eps=1e-5):
    return ms.grad(_bn_fwd, 0)(x, weight, bias, mean, var,
                               training, momentum, eps)


def _grads_all(x, weight, bias, mean, var, training=False,
               momentum=0.1, eps=1e-5):
    return ms.grad(_bn_fwd, grad_position=(0, 1, 2))(
        x, weight, bias, mean, var, training, momentum, eps)


def _set_mode(mode):
    if mode == 'kbk':
        context.set_context(mode=context.GRAPH_MODE, jit_level='O0')
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_bn_ext_bprop_dtype_align(mode):
    """
    Feature: BatchNorm bprop dtype alignment.
    Description: Verify gradients dtypes align with forward inputs when using
        mint.nn.functional.batch_norm in both PyNative and KBK modes.
    Expectation: gx/gw/gb have the same dtype as x/weight/bias respectively.
    """
    _set_mode(mode)
    np.random.seed(1)
    x = Tensor(np.random.randn(2, 4, 3, 3).astype(np.float16))
    weight = Tensor(np.random.randn(4).astype(np.float32))
    bias = Tensor(np.random.randn(4).astype(np.float32))
    mean = Tensor(np.random.randn(4).astype(np.float32))
    var = Tensor(np.abs(np.random.randn(4)).astype(np.float32))

    gx, gw, gb = _grads_all(x, weight, bias, mean, var, True, 0.1, 1e-5)
    assert gx.dtype == x.dtype
    assert gw.dtype == weight.dtype
    assert gb.dtype == bias.dtype


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_bn_ext_bprop_skip_cast(mode):
    """
    Feature: BatchNorm bprop cast guard.
    Description: Pass None for weight and bias so only grad of x is requested.
        Verify no invalid Cast is triggered in PyNative and KBK modes.
    Expectation: No exception; gx.dtype equals x.dtype.
    """
    _set_mode(mode)
    np.random.seed(2)
    x = Tensor(np.random.randn(2, 4, 3, 3).astype(np.float16))
    weight = None
    bias = None
    mean = Tensor(np.random.randn(4).astype(np.float32))
    var = Tensor(np.abs(np.random.randn(4)).astype(np.float32))

    gx = _grad_x(x, weight, bias, mean, var, True, 0.1, 1e-5)
    assert gx.dtype == x.dtype
