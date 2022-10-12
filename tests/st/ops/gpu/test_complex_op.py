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
import torch
import mindspore.ops.operations as P
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore import ops
from mindspore.ops import Complex as ComplexOp

shape_2d = (7, 6)
shape_1d = (6,)
rand_shape_1d_1 = np.random.rand(*shape_1d).astype(np.float32)
rand_shape_1d_2 = np.random.rand(*shape_1d).astype(np.float32)
rand_shape_2d_1 = np.random.rand(*shape_2d).astype(np.float32)
rand_shape_2d_2 = np.random.rand(*shape_2d).astype(np.float32)
real_op = ops.Real()
imag_op = ops.Imag()


class Complex(Cell):
    def __init__(self):
        super().__init__()
        self.complex = ComplexOp()

    def construct(self, x, y):
        return self.complex(x, y)


def complex_compare(complex1, complex2):
    real1 = real_op(Tensor(complex1)).asnumpy()
    real2 = np.real(complex2)
    imag1 = imag_op(Tensor(complex1)).asnumpy()
    imag2 = np.imag(complex2)
    return np.allclose(real1, real2, rtol=5e-03, atol=5e-03) and np.allclose(imag1, imag2, rtol=5e-03, atol=5e-03)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_complex_elemwise():
    """
    Feature: complex basic Operation.
    Description: Test complex basic Operation.
    Expectation: the result match given one.
    """

    real_ms = Tensor(rand_shape_2d_1)
    imag_ms = Tensor(rand_shape_2d_2)
    real_to = rand_shape_2d_1
    imag_to = rand_shape_2d_2

    complex1 = Complex()(real_ms, imag_ms)
    complex2 = Complex()(imag_ms, real_ms)
    complex_1 = np.vectorize(complex)(real_to, imag_to)
    complex_2 = np.vectorize(complex)(imag_to, real_to)
    assert complex_compare(complex1, complex_1)

    res_ms = ops.Add()(complex1, complex2)
    res_to = np.add(complex_1, complex_2)
    assert complex_compare(res_ms, res_to)

    res_ms = ops.Mul()(complex1, complex2)
    res_to = np.multiply(complex_1, complex_2)
    assert complex_compare(res_ms, res_to)

    res_ms = ops.Sub()(complex1, complex2)
    res_to = np.subtract(complex_1, complex_2)
    assert complex_compare(res_ms, res_to)

    res_ms = ops.Div()(complex1, complex2)
    res_to = np.divide(complex_1, complex_2)
    assert complex_compare(res_ms, res_to)

    res_ms = complex1 / complex2
    res_to = np.divide(complex_1, complex_2)
    assert complex_compare(res_ms, res_to)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_complex_broadcast():
    """
    Feature: complex broadcast Operation.
    Description: Test complex broadcast Operation.
    Expectation: the result match given one.
    """

    real_ms_1 = Tensor(rand_shape_2d_1)
    imag_ms_1 = Tensor(rand_shape_2d_2)
    real_ms_2 = Tensor(rand_shape_1d_1)
    imag_ms_2 = Tensor(rand_shape_1d_2)
    real_to_1 = rand_shape_2d_1
    imag_to_1 = rand_shape_2d_2
    real_to_2 = rand_shape_1d_1
    imag_to_2 = rand_shape_1d_2

    complex1 = Complex()(real_ms_1, imag_ms_1)
    complex2 = Complex()(real_ms_2, imag_ms_2)
    complex_1 = np.vectorize(complex)(real_to_1, imag_to_1)
    complex_2 = np.vectorize(complex)(real_to_2, imag_to_2)
    assert complex_compare(complex1, complex_1)

    res_ms = ops.Add()(complex1, complex2)
    res_to = np.add(complex_1, complex_2)
    assert complex_compare(res_ms, res_to)

    res_ms = ops.Mul()(complex1, complex2)
    res_to = np.multiply(complex_1, complex_2)
    assert complex_compare(res_ms, res_to)

    res_ms = ops.Sub()(complex1, complex2)
    res_to = np.subtract(complex_1, complex_2)
    assert complex_compare(res_ms, res_to)

    res_ms = ops.Div()(complex1, complex2)
    res_to = np.divide(complex_1, complex_2)
    assert complex_compare(res_ms, res_to)

    res_ms = complex1 / complex2
    res_to = np.divide(complex_1, complex_2)
    assert complex_compare(res_ms, res_to)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_complex_reducesum():
    """
    Feature: complex reduce Operation.
    Description: Test complex reduce Operation.
    Expectation: the result match given one.
    """
    x0_real = np.random.rand(2, 3, 4).astype(np.float32)
    x0_imag = np.random.rand(2, 3, 4).astype(np.float32)
    x0 = Complex()(Tensor(x0_real), Tensor(x0_imag))
    axis0 = 2
    keep_dims0 = True
    res_ms = P.ReduceSum(keep_dims0)(x0, axis0)
    x0_torch = torch.complex(torch.tensor(x0_real), torch.tensor(x0_imag))
    res_torch = torch.sum(x0_torch, axis0, keep_dims0)
    np.allclose(res_ms.asnumpy(), res_torch.numpy(), rtol=5e-03, atol=1.e-8)
