# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P


class MatMulNet(nn.Cell):
    def __init__(self, transpose_a=False, transpose_b=False):
        super(MatMulNet, self).__init__()
        self.matmul = P.MatMul(transpose_a, transpose_b)

    def construct(self, x, y):
        return self.matmul(x, y)

def judge_result_correct(result, expect):
    assert result.dtype == expect.dtype
    assert result.shape == expect.shape
    assert np.allclose(result, expect)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_matmul_no_transpose_vec():
    input_x = Tensor(np.arange(1 * 3).reshape((1, 3)), mstype.float32)
    input_y = Tensor(np.arange(3 * 5).reshape((3, 5)), mstype.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = MatMulNet()
    output = net(input_x, input_y)
    expect = np.array([[25., 28., 31., 34., 37.]], dtype=np.float32)
    judge_result_correct(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_matmul_no_transpose():
    input_x = Tensor(np.arange(4 * 3).reshape((4, 3)), mstype.float32)
    input_y = Tensor(np.arange(3 * 5).reshape((3, 5)), mstype.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = MatMulNet()
    output = net(input_x, input_y)
    expect = np.array([[25., 28., 31., 34., 37.],
                       [70., 82., 94., 106., 118.],
                       [115., 136., 157., 178., 199.],
                       [160., 190., 220., 250., 280.]], dtype=np.float32)
    judge_result_correct(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_matmul_transpose_a():
    input_x = Tensor(np.arange(3 * 2).reshape((3, 2)), mstype.float32)
    input_y = Tensor(np.arange(3 * 4).reshape((3, 4)), mstype.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = MatMulNet(transpose_a=True)
    output = net(input_x, input_y)
    expect = np.array([[40., 46., 52., 58.],
                       [52., 61., 70., 79.]], dtype=np.float32)
    judge_result_correct(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_matmul_transpose_b():
    input_x = Tensor(np.arange(2 * 3).reshape((2, 3)), mstype.float32)
    input_y = Tensor(np.arange(5 * 3).reshape((5, 3)), mstype.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = MatMulNet(transpose_b=True)
    output = net(input_x, input_y)
    expect = np.array([[5., 14., 23., 32., 41.],
                       [14., 50., 86., 122., 158.]], dtype=np.float32)
    judge_result_correct(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_matmul_transpose_ab():
    input_x = Tensor(np.arange(3 * 5).reshape((3, 5)), mstype.float16)
    input_y = Tensor(np.arange(4 * 3).reshape((4, 3)), mstype.float16)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = MatMulNet(transpose_a=True, transpose_b=True)
    output = net(input_x, input_y)
    expect = np.array([[25., 70., 115., 160.],
                       [28., 82., 136., 190.],
                       [31., 94., 157., 220.],
                       [34., 106., 178., 250.],
                       [37., 118., 199., 280.]], dtype=np.float16)
    judge_result_correct(output.asnumpy(), expect)
