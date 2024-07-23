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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.operations import _inner_ops as inner


class NetInv(nn.Cell):
    def __init__(self):
        super(NetInv, self).__init__()
        self.inv = P.Inv()

    def construct(self, x):
        return self.inv(x)


class InvDynamicShapeNet(nn.Cell):
    def __init__(self):
        super(InvDynamicShapeNet, self).__init__()
        self.test_dynamic = inner.GpuConvertToDynamicShape()

    def construct(self, x):
        x = self.test_dynamic(x)
        return F.inv(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6)])
@pytest.mark.parametrize('dtype, tol',
                         [(np.int32, 1.0e-4), (np.int64, 1.0e-4), (np.float16, 1.0e-3), (np.float32, 1.0e-4),
                          (np.float64, 1.0e-5), (np.complex64, 1.0e-6), (np.complex128, 1.0e-10)])
def test_inv(mode, shape, dtype, tol):
    """
    Feature: ALL To ALL
    Description: test cases for inv
    Expectation: the result match to numpy
    """
    context.set_context(mode=mode, device_target="GPU")
    inv = NetInv()
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(*shape).astype(dtype) * prop
    output = inv(Tensor(x))
    expect_output = (1.0 / x).astype(dtype)
    assert np.allclose(output.asnumpy(), expect_output, atol=tol, rtol=tol, equal_nan=True)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_inv_vmap(mode):
    """
    Feature: test inv vmap feature.
    Description: test inv vmap feature.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target="GPU")
    x = Tensor(np.array([[0.25, 0.4, 0.31, 0.52], [0.5, 0.12, 0.31, 0.58]], dtype=np.float32))
    # Case 1
    output = F.vmap(F.inv, 0, 0)(x)
    expect_output = np.array([[4., 2.5, 3.2258065, 1.923077], [2., 8.333334, 3.2258065, 1.724138]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    # Case 2
    output = F.vmap(F.inv, 1, 0)(x)
    expect_output = np.array([[4., 2.], [2.5, 8.333334], [3.2258065, 3.2258065], [1.923077, 1.724138]],
                             dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    # Case 3
    output = F.vmap(F.inv, 0, 1)(x)
    expect_output = np.array([[4., 2.], [2.5, 8.333334], [3.2258065, 3.2258065], [1.923077, 1.724138]],
                             dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_inv_dynamic_shape(mode):
    """
    Feature: test inv dynamic_shape feature.
    Description: test inv dynamic_shape feature.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target="GPU")
    x = Tensor(np.array([[0.25, 0.4, 0.31, 0.52],
                         [0.5, 0.12, 0.31, 0.58]], dtype=np.float32))
    output = InvDynamicShapeNet()(x)
    expect_output = np.array([[4., 2.5, 3.2258065, 1.923077],
                              [2., 8.333334, 3.2258065, 1.724138]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)
