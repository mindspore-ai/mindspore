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

import os

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore import dtype as mstype
from mindspore.ops.operations import _inner_ops as inner

class MatMulNet(nn.Cell):
    def __init__(self):
        super(MatMulNet, self).__init__()
        self.matmul = P.MatMul()

    def construct(self, x, y):
        return self.matmul(x, y)


class MatMul_d(nn.Cell):
    def __init__(self):
        super(MatMul_d, self).__init__()
        self.test_dynamic = inner.GpuConvertToDynamicShape()
        self.matmul = P.MatMul()

    def construct(self, x, y):
        x = self.test_dynamic(x)
        y = self.test_dynamic(y)
        return self.matmul(x, y)


class MatMulComposite(nn.Cell):
    def __init__(self):
        super(MatMulComposite, self).__init__()
        self.matmul = C.matmul

    def construct(self, x, y):
        return self.matmul(x, y)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_MatMul_dynamic():

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = MatMul_d()

    x1 = np.arange(2).reshape(1, 2).astype(np.float32)
    y1 = np.arange(4).reshape(2, 2).astype(np.float32)
    output1 = net(Tensor(x1), Tensor(y1))
    expect1 = np.matmul(x1, y1)
    np.testing.assert_array_almost_equal(output1.asnumpy(), expect1)

    x2 = np.arange(102).reshape(34, 3).astype(np.float32)
    y2 = np.arange(18).reshape(3, 6).astype(np.float32)
    output2 = net(Tensor(x2), Tensor(y2))
    expect2 = np.matmul(x2, y2)
    np.testing.assert_array_almost_equal(output2.asnumpy(), expect2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_matmul_float64():

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = MatMulNet()

    x = np.arange(102).reshape(34, 3).astype(np.float64)
    y = np.arange(18).reshape(3, 6).astype(np.float64)
    output = net(Tensor(x), Tensor(y))
    expect = np.matmul(x, y)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect)

@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_matmul_composite():

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = MatMulComposite()

    scalars = [np.random.randn(1).astype(np.float32), np.random.randn(1).astype(np.float32)]
    for x in scalars:
        for y in scalars:
            output = net(Tensor(x), Tensor(y))
            expect = np.matmul(x, y)
            np.testing.assert_array_almost_equal(output.asnumpy(), expect, decimal=4)

    broadcastables = [
        np.random.randn(5, 2).astype(np.float32), np.random.randn(2, 5).astype(np.float32),
        np.random.randn(2, 9).astype(np.float32), np.random.randn(9, 2).astype(np.float32),
        np.random.randn(9, 2).astype(np.float32), np.random.randn(2, 3).astype(np.float32),
        np.random.randn(5, 4).astype(np.float32), np.random.randn(4, 1).astype(np.float32)
    ]
    for i in range(4):
        x = broadcastables[2*i]
        y = broadcastables[2*i + 1]
        output = net(Tensor(x), Tensor(y))
        expect = np.matmul(x, y)
        np.testing.assert_array_almost_equal(output.asnumpy(), expect, decimal=4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_matmul_tensor_api_modes(mode):
    """
    Feature: Test matmul tensor api.
    Description: Test matmul tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="GPU")
    x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4), mstype.float32)
    y = Tensor(np.arange(4 * 5).reshape(4, 5), mstype.float32)
    output = x.matmul(y)
    expected = np.array([[[70., 76., 82., 88., 94.],
                          [190., 212., 234., 256., 278.],
                          [310., 348., 386., 424., 462.]],
                         [[430., 484., 538., 592., 646.],
                          [550., 620., 690., 760., 830.],
                          [670., 756., 842., 928., 1014.]]], np.float32)
    np.testing.assert_array_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_matmul_tensor_core(mode):
    """
    Feature: Test matmul tensor api.
    Description: Test matmul tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="GPU")
    m = 300
    n = 300
    k = 400
    x_np = np.random.randn(m * k).astype(np.float32)
    y_np = np.random.randn(k * n).astype(np.float32)
    x_np.shape = m, k
    y_np.shape = k, n
    x_ms = Tensor(x_np)
    y_ms = Tensor(y_np)

    context.set_context(gpu_config={"matmul_allow_tf32": False})
    out_ms_fp32 = P.MatMul()(x_ms, y_ms).asnumpy()

    context.set_context(gpu_config={"matmul_allow_tf32": True})
    out_ms_tf32 = P.MatMul()(x_ms, y_ms).asnumpy()

    assert np.abs(out_ms_fp32 - out_ms_tf32).mean() < 0.005


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_matmul_dtypes():
    """
    Feature: Test matmul dtypes.
    Description: Test matmul dtypes for Graph mode.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    m = 3
    n = 3
    k = 4
    x_np = np.random.randn(m * k).astype(np.float32)
    y_np = np.random.randn(k * n).astype(np.float32)
    x_np.shape = m, k
    y_np.shape = k, n
    matmul = MatMulNet()
    valid_dtypes = (mstype.float16, mstype.float32, mstype.float64, mstype.complex64, mstype.complex128)
    all_dtypes = mstype.all_types
    for dtype in all_dtypes:
        # bfloat16 is not supported yet
        if dtype == mstype.bfloat16:
            continue
        x_ms = Tensor(x_np).astype(dtype)
        y_ms = Tensor(y_np).astype(dtype)
        os.environ['MS_DISABLE_KERNEL_BACKOFF'] = '1'
        if dtype in valid_dtypes:
            out = matmul(x_ms, y_ms)
            assert out.dtype == x_ms.dtype
        else:
            with pytest.raises(TypeError):
                matmul(x_ms, y_ms)
        del os.environ['MS_DISABLE_KERNEL_BACKOFF']


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_matmul_fp16():
    """
    Feature: Test matmul fp16 results.
    Description: Test matmul fp16 for Graph mode.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_np = np.random.randn(256, 512).astype(np.float16)
    y_np = np.random.randn(512, 1024).astype(np.float16)
    matmul = P.MatMul()
    x_ms = Tensor(x_np)
    y_ms = Tensor(y_np)
    out_ms = matmul(x_ms, y_ms).asnumpy()
    out_np = np.matmul(x_np, y_np)
    assert np.abs(out_ms - out_np).mean() < 1e-4
    assert out_ms.dtype == np.float16
