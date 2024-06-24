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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations.math_ops import LuUnpack
import mindspore.common.dtype as mstype
import torch
import pytest


class LuUnpackNet(nn.Cell):
    def __init__(self, unpack_data=True, unpack_pivots=True):
        super(LuUnpackNet, self).__init__()
        self.lu_unpack = LuUnpack(unpack_data=unpack_data, unpack_pivots=unpack_pivots)

    def construct(self, LU_data, LU_pivots):
        return self.lu_unpack(LU_data, LU_pivots)


@pytest.mark.skip(reason="never run on ci or smoke test")
def test_lu_unpack_float32_int64():
    """
    Feature: ALL To ALL
    Description: test cases for LuUnpack
    Expectation: the result match to torch
    """
    my_atol = 1e-5
    data = torch.randn(10, 7, 8, 9)
    a_lu, pivots = data.lu()
    p, a_l, a_u = torch.lu_unpack(a_lu, pivots)
    my_a_lu = Tensor(a_lu.numpy(), mstype.float32)
    my_pivots = Tensor(pivots.numpy(), mstype.int64)
    net = LuUnpackNet(unpack_data=True, unpack_pivots=True)
    my_p, my_a_l, my_a_u = net(my_a_lu, my_pivots)
    assert np.allclose(p.numpy(), my_p.asnumpy(), atol=my_atol)
    assert np.allclose(a_l.numpy(), my_a_l.asnumpy(), atol=my_atol)
    assert np.allclose(a_u.numpy(), my_a_u.asnumpy(), atol=my_atol)


@pytest.mark.skip(reason="never run on ci or smoke test")
def test_lu_unpack_float64_int32():
    """
    Feature: ALL To ALL
    Description: test cases for LuUnpack
    Expectation: the result match to torch
    """
    my_atol = 1e-5
    data = torch.randn(7, 8, 9, 10)
    a_lu, pivots = data.lu()
    p, a_l, a_u = torch.lu_unpack(a_lu, pivots)
    my_a_lu = Tensor(a_lu.numpy(), mstype.float64)
    my_pivots = Tensor(pivots.numpy(), mstype.int32)
    net = LuUnpackNet(unpack_data=True, unpack_pivots=True)
    my_p, my_a_l, my_a_u = net(my_a_lu, my_pivots)
    assert np.allclose(p.numpy(), my_p.asnumpy(), atol=my_atol)
    assert np.allclose(a_l.numpy(), my_a_l.asnumpy(), atol=my_atol)
    assert np.allclose(a_u.numpy(), my_a_u.asnumpy(), atol=my_atol)


@pytest.mark.skip(reason="never run on ci or smoke test")
def test_lu_unpack_float64_int16():
    """
    Feature: ALL To ALL
    Description: test cases for LuUnpack
    Expectation: the result match to torch
    """
    my_atol = 1e-5
    data = torch.randn(4, 6, 8, 10)
    a_lu, pivots = data.lu()
    p, a_l, a_u = torch.lu_unpack(a_lu, pivots)
    my_a_lu = Tensor(a_lu.numpy(), mstype.float64)
    my_pivots = Tensor(pivots.numpy(), mstype.int16)
    net = LuUnpackNet(unpack_data=True, unpack_pivots=True)
    my_p, my_a_l, my_a_u = net(my_a_lu, my_pivots)
    assert np.allclose(p.numpy(), my_p.asnumpy(), atol=my_atol)
    assert np.allclose(a_l.numpy(), my_a_l.asnumpy(), atol=my_atol)
    assert np.allclose(a_u.numpy(), my_a_u.asnumpy(), atol=my_atol)


@pytest.mark.skip(reason="never run on ci or smoke test")
def test_lu_unpack_input_error():
    """
    Feature: ALL To ALL
    Description: test cases for LuUnpack
    Expectation: raise ValueError
    """
    my_a_lu = Tensor(np.random.randn(3, 3, 2, 2).astype(np.float32))
    my_pivots = Tensor(np.random.randn(3, 3, 5, 3).astype(np.int32))
    with pytest.raises(ValueError):
        net = LuUnpackNet(unpack_data=True, unpack_pivots=True)
        net(my_a_lu, my_pivots)
