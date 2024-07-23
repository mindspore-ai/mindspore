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

import mindspore.context as context
from mindspore import Tensor
import mindspore.ops.operations._grad_ops as P
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
np.random.seed(1)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_atangrad_fp32():
    """
    Feature: ALL To ALL
    Description: test cases for AtanGrad float32
    Expectation: the result match to numpy
    """
    x_np = np.random.rand(4, 2).astype(np.float32) * 10
    dout_np = np.random.rand(4, 2).astype(np.float32) * 10
    output_ms = P.AtanGrad()(Tensor(x_np), Tensor(dout_np))
    output_np = dout_np / (1 + np.square(x_np))
    assert np.allclose(output_ms.asnumpy(), output_np, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_atangrad_fp16():
    """
    Feature: ALL To ALL
    Description: test cases for AtanGrad float16
    Expectation: the result match to numpy
    """
    x_np = np.random.rand(4, 2).astype(np.float16) * 10
    dout_np = np.random.rand(4, 2).astype(np.float16) * 10
    output_ms = P.AtanGrad()(Tensor(x_np), Tensor(dout_np))
    output_np = dout_np.astype(np.float32) / (1 + np.square(x_np.astype(np.float32)))
    assert np.allclose(output_ms.asnumpy(), output_np.astype(np.float16), 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_atan_grad_float(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for AtanGrad
    Expectation: the result match to numpy
    """
    x = np.array([-0.5, 0, 0.5]).astype(dtype)
    dy = np.array([1, 0, -1]).astype(dtype)
    output = P.AtanGrad()(Tensor(x), Tensor(dy))
    print(output)
    expect = dy / (1 + x * x)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_atan_grad_complex(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for AtanGrad
    Expectation: the result match to numpy
    """
    x = np.array([-0.5, 0, 0.5]).astype(dtype)
    x = x + 0.5j * x
    dy = np.array([1, 0, -1]).astype(dtype)
    dy = dy + 0.3j * dy
    output = P.AtanGrad()(Tensor(x), Tensor(dy))
    print(output)
    expect = dy / np.conjugate(1 + x * x)
    assert np.allclose(output.asnumpy(), expect)
