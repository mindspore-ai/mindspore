# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
import scipy as scp
import mindspore.nn as nn
import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops.operations import math_ops as P


class LuNet(nn.Cell):
    def __init__(self, output_idx_type=mstype.int32):
        super(LuNet, self).__init__()
        self.lu = P.Lu(output_idx_type=output_idx_type)

    def construct(self, a):
        return self.lu(a)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_lu_dtype_float32():
    """
    Feature: Lu gpu TEST.
    Description: float32 test case for Lu
    Expectation: the result match to scp
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_np = np.array([[2.5, 3.1, 3.5], [4.7, 1.9, 0.2], [1.1, 3.6, 2.0]])
    expect, _ = scp.linalg.lu_factor(x_np)
    input_x = Tensor(x_np.astype(np.float32))
    net = LuNet(mstype.int32)
    lu, _ = net(input_x)
    rtol = 1.e-4
    atol = 1.e-4
    assert np.allclose(expect, lu.asnumpy(), rtol=rtol, atol=atol)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_lu_dtype_float64():
    """
    Feature: Lu gpu TEST.
    Description: float64 test case for Lu
    Expectation: the result match to scp
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_np = np.array([[3.5, 6.5, 3.1], [4.7, 1.9, 6.2], [1.5, 4.8, 2.3]])
    expect, _ = scp.linalg.lu_factor(x_np)
    input_x = Tensor(x_np.astype(np.float64))
    net = LuNet(mstype.int64)
    lu, _ = net(input_x)
    rtol = 1.e-5
    atol = 1.e-5
    assert np.allclose(expect, lu.asnumpy(), rtol=rtol, atol=atol)
