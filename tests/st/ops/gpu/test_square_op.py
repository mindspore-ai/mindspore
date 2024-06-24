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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class SquareNet(nn.Cell):
    def __init__(self):
        super(SquareNet, self).__init__()
        self.ops = P.Square()

    def construct(self, x):
        return self.ops(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("dtype", [np.bool_, np.int8, np.int16, np.int32, np.int64,
                                   np.uint8, np.uint16, np.uint32, np.uint64,
                                   np.float16, np.float32, np.float64,
                                   np.complex64, np.complex128])
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_square_normal(dtype, mode):
    """
    Feature: ALL To ALL
    Description: test cases for Square
    Expectation: the result match to numpy
    """
    context.set_context(mode=mode, device_target="GPU")
    error_tol = 1.0e-3
    x_np = np.random.rand(2, 3, 4, 4).astype(dtype)
    output_ms = P.Square()(Tensor(x_np))
    output_np = np.square(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np, error_tol, error_tol)
    x_np = np.random.rand(2, 3, 1, 5, 4, 4).astype(dtype)
    output_ms = P.Square()(Tensor(x_np))
    output_np = np.square(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np, error_tol, error_tol)
    x_np = np.random.rand(2).astype(dtype)
    output_ms = P.Square()(Tensor(x_np))
    output_np = np.square(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np, error_tol, error_tol)
