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
import mindspore as ms
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F


class OpNetWrapperBitwise(nn.Cell):
    """OpNetWrapperBitwise"""

    def __init__(self, op):
        """__init__"""
        super(OpNetWrapperBitwise, self).__init__()
        self.op = op

    def construct(self, *inputs):
        """construct"""
        return self.op(*inputs)


suport_type_list = [np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
mode_list = [context.PYNATIVE_MODE, context.GRAPH_MODE]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6), (3, 4, 5, 6, 2)])
@pytest.mark.parametrize('dtype', suport_type_list)
@pytest.mark.parametrize('mode', mode_list)
def test_bitwise_and(shape, dtype, mode):
    """
    Feature: BitwiseAnd gpu kernel.
    Description: test the rightness of BitwiseAnd gpu kernel.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target='GPU')
    op = P.BitwiseAnd()
    op_wrapper = OpNetWrapperBitwise(op)

    prop = 100 if np.random.random() > 0.5 else -100
    x_np = (np.random.randn(*shape) * prop).astype(dtype)
    y_np = (np.random.randn(*shape) * prop).astype(dtype)
    outputs = op_wrapper(Tensor(x_np), Tensor(y_np))
    outputs_func = F.bitwise_and(Tensor(x_np), Tensor(y_np))
    expect = np.bitwise_and(x_np, y_np)

    assert np.allclose(outputs.asnumpy(), expect)
    assert np.allclose(outputs_func.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6), (3, 4, 5, 6, 2)])
@pytest.mark.parametrize('dtype', suport_type_list)
@pytest.mark.parametrize('mode', mode_list)
def test_bitwise_or(shape, dtype, mode):
    """
    Feature: BitwiseOr gpu kernel.
    Description: test the rightness of BitwiseOr gpu kernel.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target='GPU')
    op = P.BitwiseOr()
    op_wrapper = OpNetWrapperBitwise(op)

    prop = 100 if np.random.random() > 0.5 else -100
    x_np = (np.random.randn(*shape) * prop).astype(dtype)
    y_np = (np.random.randn(*shape) * prop).astype(dtype)
    outputs = op_wrapper(Tensor(x_np), Tensor(y_np))
    outputs_func = F.bitwise_or(Tensor(x_np), Tensor(y_np))
    expect = np.bitwise_or(x_np, y_np)

    assert np.allclose(outputs.asnumpy(), expect)
    assert np.allclose(outputs_func.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6), (3, 4, 5, 6, 2)])
@pytest.mark.parametrize('dtype', suport_type_list)
@pytest.mark.parametrize('mode', mode_list)
def test_bitwise_xor(shape, dtype, mode):
    """
    Feature: BitwiseXor gpu kernel.
    Description: test the rightness of BitwiseXor gpu kernel.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target='GPU')
    op = P.BitwiseXor()
    op_wrapper = OpNetWrapperBitwise(op)

    prop = 100 if np.random.random() > 0.5 else -100
    x_np = (np.random.randn(*shape) * prop).astype(dtype)
    y_np = (np.random.randn(*shape) * prop).astype(dtype)
    outputs = op_wrapper(Tensor(x_np), Tensor(y_np))
    outputs_func = F.bitwise_xor(Tensor(x_np), Tensor(y_np))
    expect = np.bitwise_xor(x_np, y_np)

    assert np.allclose(outputs.asnumpy(), expect)
    assert np.allclose(outputs_func.asnumpy(), expect)


class NetBitwiseGPU(nn.Cell):
    """NetBitwiseGPU"""

    def construct(self, input_x, input_y):
        """construct"""
        out_and = input_x & input_y
        out_or = input_x | input_y
        out_xor = input_x ^ input_y
        return out_and, out_or, out_xor


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', mode_list)
def test_bitwise_bool(mode):
    """
    Feature: Bitwise gpu kernel.
    Description: test the rightness of Bitwise cpu kernel tensor operations.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target='CPU')

    input_x = ms.Tensor([True, False], dtype=ms.bool_)
    input_y = ms.Tensor([True, True], dtype=ms.bool_)

    net = NetBitwiseGPU()
    out = net(input_x, input_y)
    expect_and_gpu = np.array([True, False])
    expect_or_gpu = np.array([True, True])
    expect_xor_gpu = np.array([False, True])
    assert np.allclose(out[0].asnumpy(), expect_and_gpu)
    assert np.allclose(out[1].asnumpy(), expect_or_gpu)
    assert np.allclose(out[2].asnumpy(), expect_xor_gpu)

    res_and = input_x & input_y
    res_or = input_x | input_y
    res_xor = input_x ^ input_y
    assert np.allclose(res_and.asnumpy(), expect_and_gpu)
    assert np.allclose(res_or.asnumpy(), expect_or_gpu)
    assert np.allclose(res_xor.asnumpy(), expect_xor_gpu)
