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
import torch

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G
import mindspore.common.dtype as mstype


class NetLuUnpackGrad(nn.Cell):
    def __init__(self, l_grad_flag=True, u_grad_flag=True):
        super().__init__()
        self.lu_unpack_grad = G.LuUnpackGrad(L_grad_flag=l_grad_flag, U_grad_flag=u_grad_flag)

    def construct(self, l_grad, u_grad, lu_data):
        return self.lu_unpack_grad(l_grad, u_grad, lu_data)


def getl(a_lu, pivots):
    a_lu_float = a_lu.float()
    a_lu_torch = torch.tensor(a_lu_float, requires_grad=True)
    out_torch = torch.lu_unpack(a_lu_torch, pivots)
    out_torch[1].backward(gradient=out_torch[1])
    return a_lu_torch.grad


def getu(a_lu, pivots):
    a_lu_float = a_lu.float()
    a_lu_torch = torch.tensor(a_lu_float, requires_grad=True)
    out_torch = torch.lu_unpack(a_lu_torch, pivots)
    out_torch[2].backward(gradient=out_torch[2])
    return a_lu_torch.grad


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_lu_unpack_grad_graph_float():
    """
    Feature: LuUnpackGrad gpu TEST.
    Description: 4d - float32 test case for LuUnpackGrad
    Expectation: the result match to numpy
    """
    a = torch.randn(3, 3, 3, 3)
    loss = 1e-5
    a_lu, pivots = a.lu()
    out = torch.lu_unpack(a_lu, pivots)

    a_l_torch = getl(a_lu, pivots)
    a_u_torch = getu(a_lu, pivots)

    a_l_mindspore = Tensor(out[1].numpy(), mstype.float32)
    a_u_mindspore = Tensor(out[2].numpy(), mstype.float32)
    a_lu_mindspore = Tensor(a_lu.numpy(), mstype.float32)

    net = NetLuUnpackGrad(l_grad_flag=True, u_grad_flag=True)
    l_grad_mindspore, u_grad_mindspore = net(a_l_mindspore, a_u_mindspore, a_lu_mindspore)

    assert np.allclose(a_l_torch, torch.tensor(l_grad_mindspore.asnumpy()), atol=loss)
    assert np.allclose(a_u_torch, torch.tensor(u_grad_mindspore.asnumpy()), atol=loss)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_lu_unpack_grad_pynative_float():
    """
    Feature: LuUnpackGrad gpu TEST.
    Description: 4d - float32 test case for LuUnpackGrad
    Expectation: the result match to numpy
    """
    a = torch.randn(5, 4, 3, 2)
    loss = 1e-5
    a_lu, pivots = a.lu()
    out = torch.lu_unpack(a_lu, pivots)

    a_l_torch = getl(a_lu, pivots)
    a_u_torch = getu(a_lu, pivots)

    a_l_mindspore = Tensor(out[1].numpy(), mstype.float32)
    a_u_mindspore = Tensor(out[2].numpy(), mstype.float32)
    a_lu_mindspore = Tensor(a_lu.numpy(), mstype.float32)

    net = NetLuUnpackGrad(l_grad_flag=True, u_grad_flag=True)
    l_grad_mindspore, u_grad_mindspore = net(a_l_mindspore, a_u_mindspore, a_lu_mindspore)

    assert np.allclose(a_l_torch, torch.tensor(l_grad_mindspore.asnumpy()), atol=loss)
    assert np.allclose(a_u_torch, torch.tensor(u_grad_mindspore.asnumpy()), atol=loss)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_lu_unpack_grad_pynative_float_error():
    """
    Feature: LuUnpackGrad gpu TEST.
    Description: 4d - uint16 test the unsupported type for LuUnpackGrad
    Expectation: the result match to numpy
    """
    a = torch.randn(8, 7, 4, 2)
    a_lu, pivots = a.lu()
    out = torch.lu_unpack(a_lu, pivots)

    a_l_mindspore = Tensor(out[1].numpy(), mstype.uint16)
    a_u_mindspore = Tensor(out[2].numpy(), mstype.uint16)
    a_lu_mindspore = Tensor(a_lu.numpy(), mstype.uint16)

    with pytest.raises(TypeError):
        net = NetLuUnpackGrad(l_grad_flag=True, u_grad_flag=True)
        net(a_l_mindspore, a_u_mindspore, a_lu_mindspore)
