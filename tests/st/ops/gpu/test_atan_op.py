# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
np.random.seed(1)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_atan_fp32():
    """
    Feature: ALL To ALL
    Description: test cases for Atan float32
    Expectation: the result match to numpy
    """
    x_np = np.random.rand(4, 2).astype(np.float32) * 10
    output_ms = P.Atan()(Tensor(x_np))
    output_np = np.arctan(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_atan_fp16():
    """
    Feature: ALL To ALL
    Description: test cases for Atan float16
    Expectation: the result match to numpy
    """
    x_np = np.random.rand(4, 2).astype(np.float16) * 10
    output_ms = P.Atan()(Tensor(x_np))
    output_np = np.arctan(x_np.astype(np.float32)).astype(np.float16)
    assert np.allclose(output_ms.asnumpy(), output_np, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_atan_float(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for Atan
    Expectation: the result match to numpy
    """
    np_array = np.array([-1, -0.5, 0, 0.5, 1], dtype=dtype)
    input_x = Tensor(np_array)
    output = P.Atan()(input_x)
    print(output)
    expect = np.arctan(np_array)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_atan_complex(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for Atan
    Expectation: the result match to numpy
    """
    np_array = np.array([-1, -0.5, 0, 0.5, 1], dtype=dtype)
    np_array = np_array + 0.5j * np_array
    input_x = Tensor(np_array)
    output = P.Atan()(input_x)
    print(output)
    expect = np.arctan(np_array)
    assert np.allclose(output.asnumpy(), expect)


def test_atan_forward_tensor_api(nptype):
    """
    Feature: test atan forward tensor api for given input dtype.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([1.0, 0.0]).astype(nptype))
    output = x.atan()
    expected = np.array([0.7853982, 0.0]).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_atan_forward_float32_tensor_api():
    """
    Feature: test atan forward tensor api.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_atan_forward_tensor_api(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    test_atan_forward_tensor_api(np.float32)
