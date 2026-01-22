# Copyright 2019-2023 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, JitConfig
from mindspore.graph.api import jit
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.graph.api import _pynative_executor

from tests.st.utils import test_utils
from tests.mark_utils import arg_mark

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = P.MatMul()

    @jit
    def construct(self, x1_, x2_):
        return self.matmul(x1_, x2_)


x1 = np.random.randn(1, 3).astype(np.float32)
x2 = np.random.randn(3, 4).astype(np.float32)


def test_net():
    matmul = Net()
    output = matmul(Tensor(x1), Tensor(x2))
    print(x1)
    print(x2)
    print(output.asnumpy())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_net_bf16(mode):
    """
    Feature: Test matmul bfloat16.
    Description: Test matmul tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="Ascend")
    x = Tensor(np.arange(1 * 3).reshape(1, 3), mstype.bfloat16)
    y = Tensor(np.arange(3 * 4).reshape(3, 4), mstype.bfloat16)
    matmul = Net()
    output = matmul(x, y)
    except_out = np.array([20., 23., 26., 29.], np.float32)
    assert np.allclose(output.float().asnumpy(), except_out, 0.004, 0.004)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_matmul_tensor_api_modes(mode):
    """
    Feature: Test matmul tensor api.
    Description: Test matmul tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="Ascend")
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


def do_test_matmul_dtypes(valid_dtypes, is_ge_only=False):
    """
    Feature: Test matmul dtypes.
    Description: Test matmul dtypes for Graph mode.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    m = 3
    n = 3
    k = 4
    x_np = np.random.randn(m * k).astype(np.float32)
    y_np = np.random.randn(k * n).astype(np.float32)
    x_np.shape = m, k
    y_np.shape = k, n
    matmul = Net()
    if is_ge_only:
        matmul.set_jit_config(JitConfig(backend="GE"))
    else:
        matmul.set_jit_config(JitConfig(jit_level="O0"))
    all_dtypes = mstype.all_types
    for dtype in all_dtypes:
        # qint4x2/bfloat16/float8 is not supported yet
        if dtype in (mstype.qint4x2, mstype.bfloat16, mstype.float8_e4m3fn, mstype.float8_e5m2, mstype.hifloat8):
            continue
        x_ms = Tensor(x_np).astype(dtype)
        y_ms = Tensor(y_np).astype(dtype)
        if dtype in valid_dtypes:
            out = matmul(x_ms, y_ms)
            if x_ms.dtype == mstype.int8:
                assert out.dtype == mstype.int32
            else:
                assert out.dtype == x_ms.dtype
        else:
            with pytest.raises((RuntimeError, TypeError)):
                matmul(x_ms, y_ms)
                _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_matmul_dtypes():
    """
    Feature: Test matmul dtypes.
    Description: Test matmul dtypes for Graph mode.
    Expectation: The result match to the expect value.
    """
    do_test_matmul_dtypes([mstype.float16, mstype.float32])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_matmul_dtypes_ge():
    """
    Feature: Test matmul dtypes.
    Description: Test matmul dtypes for Graph mode.
    Expectation: The result match to the expect value.
    """
    do_test_matmul_dtypes([mstype.int8, mstype.int32, mstype.float16, mstype.float32], True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trans_double_matmul():
    """
    Feature: Test transpose and double matmul.
    Description: Test matmul success for Graph mode.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, jit_level='O0')

    class TransDoubleMatMulNet(nn.Cell):
        def __init__(self, trans_a1=False, trans_b1=False, trans_a2=False, trans_b2=False):
            super(TransDoubleMatMulNet, self).__init__()
            self.mm1 = P.MatMul(trans_a1, trans_b1)
            self.mm2 = P.MatMul(trans_a2, trans_b2)
            self.trans = P.Transpose()

        def construct(self, x, y1, y2, perm):
            x = self.trans(x, perm)
            out1 = self.mm1(x, y1)
            out2 = self.mm2(x, y2)
            return out1, out2

    x = Tensor(np.random.randn(8, 96), dtype=mstype.float16)
    y1 = Tensor(np.random.randn(96, 24), dtype=mstype.float16)
    y2 = Tensor(np.random.randn(8, 24), dtype=mstype.float16)

    net = TransDoubleMatMulNet(True, False, False, False)
    out1, out2 = net(x, y1, y2, (1, 0))

    assert out1.shape == (8, 24)
    assert out2.shape == (96, 24)
