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
import numpy as np
import scipy as scp
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.ops import PrimitiveWithInfer
from mindspore.ops import prim_attr_register
from mindspore._checkparam import Validator as validator

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class NetCholesky(nn.Cell):
    def __init__(self):
        super(NetCholesky, self).__init__()
        self.cholesky = P.Cholesky()

    def construct(self, x):
        return self.cholesky(x)


class ScipyCholesky(PrimitiveWithInfer):
    """
    Inner API for Cholesky base class.
    """

    @prim_attr_register
    def __init__(self, lower=False, clean=False):
        super().__init__(name="PureCholesky")
        self.lower = validator.check_value_type("lower", lower, [bool], self.lower)
        self.clean = validator.check_value_type("clean", clean, [bool], self.clean)
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def __infer__(self, x):
        x_shape = x['shape']
        x_dtype = x['dtype']
        return {
            'shape': tuple(x_shape),
            'dtype': x_dtype,
            'value': None
        }


class ScipyNetCholesky(nn.Cell):
    def __init__(self, lower=False, clean=False):
        super(ScipyNetCholesky, self).__init__()
        self.cholesky = ScipyCholesky(lower, clean)

    def construct(self, x):
        return self.cholesky(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cholesky_fp32():
    """
    Feature: ALL TO ALL
    Description:  test cases for origin cholesky [N,N]
    Expectation: the result match np cholesky
    """
    cholesky = NetCholesky()
    x = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]]).astype(np.float32)
    output = cholesky(Tensor(x, dtype=mstype.float32))
    expect = np.linalg.cholesky(x)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect) < tol).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scipy_cholesky_fp32():
    """
    Feature: ALL TO ALL
    Description:  test cases for new scipy cholesky [N,N]
    Expectation: the result match scipy cholesky
    """
    a = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]]).astype(np.float32)
    tensor_a = Tensor(a)
    cholesky = ScipyNetCholesky(lower=True, clean=False)
    output = cholesky(tensor_a)

    cholesky1 = ScipyNetCholesky(lower=False, clean=False)
    output1 = cholesky1(tensor_a)

    expect = scp.linalg.cholesky(a, lower=True)
    expect1 = scp.linalg.cholesky(a, lower=False)

    rtol = 1.e-4
    atol = 1.e-5
    assert np.allclose(expect, output.asnumpy(), rtol=rtol, atol=atol)
    assert np.allclose(expect1, output1.asnumpy(), rtol=rtol, atol=atol)
