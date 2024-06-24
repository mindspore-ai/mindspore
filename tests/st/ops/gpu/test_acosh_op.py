# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore.ops import operations as P
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_acosh_fp32():
    """
    Feature: acosh kernel
    Description: test acosh float32
    Expectation: just test
    """
    x_np = np.random.rand(4, 2).astype(np.float32) * 10 + 1
    output_ms = P.Acosh()(Tensor(x_np))
    output_np = np.arccosh(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_acosh_fp16():
    """
    Feature: acosh kernel
    Description: test acosh float16
    Expectation: just test
    """
    x_np = np.random.rand(4, 2).astype(np.float16) * 10 + 1
    output_ms = P.Acosh()(Tensor(x_np))
    output_np = np.arccosh(x_np.astype(np.float32)).astype(np.float16)
    assert np.allclose(output_ms.asnumpy(), output_np, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_acosh_fp64():
    """
    Feature: acosh kernel
    Description: test acosh float64
    Expectation: just test
    """
    x_np = np.random.rand(4, 2).astype(np.float64) * 10 + 1
    output_ms = P.Acosh()(Tensor(x_np))
    output_np = np.arccosh(x_np.astype(np.float64)).astype(np.float64)
    assert np.allclose(output_ms.asnumpy(), output_np, 1e-6, 1e-6)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_acosh_complex64():
    """
    Feature: acosh kernel
    Description: test acosh complex64
    Expectation: just test
    """
    x_np = np.random.rand(4, 2).astype(np.complex64) * 10 + 1
    x_np = x_np + 2j*x_np
    output_ms = P.Acosh()(Tensor(x_np))
    output_np = np.arccosh(x_np.astype(np.complex64)).astype(np.complex64)
    assert np.allclose(output_ms.asnumpy(), output_np, 1e-6, 1e-6)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_acosh_complex128():
    """
    Feature: acosh kernel
    Description: test acosh complex128
    Expectation: just test
    """
    x_np = np.random.rand(4, 2).astype(np.complex128) * 10 + 1
    x_np = x_np + 2j*x_np
    output_ms = P.Acosh()(Tensor(x_np))
    output_np = np.arccosh(x_np.astype(np.complex128)).astype(np.complex128)
    assert np.allclose(output_ms.asnumpy(), output_np, 1e-12, 1e-12)
