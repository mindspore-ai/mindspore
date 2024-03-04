# Copyright 2024 Huawei Technologies Co., Ltd
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
import os

import numpy as np
import pytest

import mindspore as ms
from mindspore import ops
from mindspore.nn import Cell
from mindspore.ops.extend import sub

rtol = 1e-3


class SubCell(Cell):
    def __init__(self):
        super().__init__()
        self.sub = sub

    def construct(self, x, y, alpha):
        return self.sub(x, y, alpha)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_forward(context_mode):
    """
    Feature: test sub forward
    Description: test sub forward
    Expectation: success
    """
    os.environ["GRAPH_OP_RUN"] = "1"
    ms.context.set_context(mode=context_mode)

    sub_cell = SubCell()

    # 2 x 2
    x = np.array([[1, 2], [3, 4]], np.float32)
    y = np.array([[1, 2], [3, 4]], np.float32)
    alpha = 2.0

    output = sub_cell(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = x - y * alpha

    np.testing.assert_allclose(output, expect, rtol=rtol)

    sub_cell.set_inputs(ms.tensor(shape=[None, None], dtype=ms.float32),
                        ms.tensor(shape=[None, None], dtype=ms.float32), alpha)

    # 3 x 3
    x = np.array([[1, 2], [3, 4]], np.float32)
    y = np.array([[1, 2], [3, 4]], np.float32)
    alpha = 2.0

    output = sub_cell(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = x - y * alpha

    np.testing.assert_allclose(output, expect, rtol=rtol)

    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.float32)
    y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.float32)

    output = sub_cell(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = x - y * alpha

    np.testing.assert_allclose(output, expect, rtol=rtol)

    sub_cell.set_inputs(ms.tensor(shape=None, dtype=ms.float32),
                        ms.tensor(shape=None, dtype=ms.float32), alpha)

    # 2 x 2 x 2
    x = np.array([[1, 2], [3, 4]], np.float32)
    y = np.array([[1, 2], [3, 4]], np.float32)
    alpha = 2.0

    output = sub_cell(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = x - y * alpha

    np.testing.assert_allclose(output, expect, rtol=rtol)

    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], np.float32)
    y = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], np.float32)

    output = sub_cell(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = x - y * alpha

    np.testing.assert_allclose(output, expect, rtol=rtol)
    del os.environ["GRAPH_OP_RUN"]


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_backward(context_mode):
    """
    Feature: test sub backward
    Description: test sub backward
    Expectation: success
    """
    os.environ["GRAPH_OP_RUN"] = "1"
    ms.context.set_context(mode=context_mode)

    sub_cell = SubCell()

    # 2 x 2
    x = np.array([[1, 2], [3, 4]], np.float32)
    y = np.array([[1, 2], [3, 4]], np.float32)
    alpha = 2.0

    output = ops.grad(sub_cell, (0))(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = np.ones_like(y)

    np.testing.assert_allclose(output, expect, rtol=rtol)
    del os.environ["GRAPH_OP_RUN"]
