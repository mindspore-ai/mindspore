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
import mindspore
import mindspore.context as context
from mindspore import Tensor, nn
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype


# all cases tested against dchip


class BatchMatMulCell(nn.Cell):
    def __init__(self):
        super().__init__()
        self.bmm = F.bmm

    def construct(self, x, y):
        return self.bmm(x, y)


def test_bmm_forward_tensor_api(nptype):
    """
    Feature: test bmm forward tensor api for given input dtype.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.ones(shape=[2, 4, 1, 3]).astype(nptype))
    y = Tensor(np.ones(shape=[2, 4, 3, 4]).astype(nptype))
    output = x.bmm(y)
    expected = 3 * np.ones(shape=[2, 4, 1, 4]).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
def test_bmm_forward_float32_tensor_api():
    """
    Feature: test bmm forward tensor api.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_bmm_forward_tensor_api(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    test_bmm_forward_tensor_api(np.float32)


def test_bmm_forward_functional_api(nptype):
    """
    Feature: test bmm forward functional api for given input dtype.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.ones(shape=[2, 4, 1, 3]).astype(nptype))
    y = Tensor(np.ones(shape=[2, 4, 3, 4]).astype(nptype))
    output = F.bmm(x, y)
    expected = 3 * np.ones(shape=[2, 4, 1, 4]).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_bmm_forward_float32_functional_api():
    """
    Feature: test bmm forward functional api.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_bmm_forward_functional_api(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    test_bmm_forward_functional_api(np.float32)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_bmm_forward_int8_functional_api():
    """
    Feature: test bmm forward functional api.
    Description: test int8 inputs.
    Expectation: the result match with expected result.
    """
    mindspore.set_context(mode=mindspore.GRAPH_MODE, jit_config={'jit_level': 'O2'})
    x = Tensor(np.ones(shape=[2, 4, 1, 3]).astype(np.int8))
    y = Tensor(np.ones(shape=[2, 4, 3, 4]).astype(np.int8))
    net = BatchMatMulCell()
    output = net(x, y)
    assert output.dtype == mstype.int32
    expected = 3 * np.ones(shape=[2, 4, 1, 4]).astype(np.int32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2.0)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_bmm_forward_functional_api_bf16(mode):
    """
    Feature: test bmm forward functional api.
    Description: test bfloat16 inputs.
    Expectation: the result match with expected result.
    """

    context.set_context(mode=mode, device_target="Ascend")
    x = Tensor(np.ones(shape=[2, 4, 1, 3]), dtype=mstype.bfloat16)
    y = Tensor(np.ones(shape=[2, 4, 3, 4]), dtype=mstype.bfloat16)
    output = F.bmm(x, y)
    expected = 3 * np.ones(shape=[2, 4, 1, 4]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.float().asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.01)
def test_batch_matmul_broadcast():
    """
    Feature: test BatchMatMul support broadcast in ascend
    Description: test BatchMatMul support broadcast.
    Expectation: no raise.
    """
    net = BatchMatMulCell()
    for shape1, shape2 in [[[3, 1, 5], [1, 5, 4]],
                           [[2, 1, 1, 5], [1, 2, 5, 4]],
                           [[3, 1, 5], [5, 4]],
                           [[2, 2, 1, 1, 5], [1, 2, 5, 4]],
                           [[3, 1, 5], [1, 3, 5, 4]]]:
        x_np = np.ones(shape=shape1)
        y_np = np.ones(shape=shape2)
        x = Tensor(x_np, dtype=mstype.float16)
        y = Tensor(y_np, dtype=mstype.float16)
        z = net(x, y)
        expected = x_np @ y_np
        np.testing.assert_array_almost_equal(z.asnumpy(), expected)
