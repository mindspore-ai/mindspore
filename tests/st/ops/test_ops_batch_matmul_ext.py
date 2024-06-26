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

import numpy as np
import pytest

import mindspore as ms
from mindspore.nn import Cell
from mindspore.ops.extend import bmm
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.pynative.utils import allclose_nparray
from tests.mark_utils import arg_mark

loss = 1e-3


class BmmCell(Cell):
    def __init__(self):
        super().__init__()
        self.bmm = bmm

    def construct(self, x, y):
        return self.bmm(x, y)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@pytest.mark.parametrize("shape1, shape2", [
    [[10, 10, 10], [10, 10, 10]],
])
def test_ops(context_mode, shape1, shape2):
    """
    Feature: BatchMatMulExt op.
    Description: test bmm_ext
    Expectation: expect correct shape result.
    """
    ms.set_context(mode=context_mode)

    bmm_cell = BmmCell()
    # 2 x 2
    x = random_input(shape1)
    y = random_input(shape2)

    output = bmm_cell(ms.tensor(x), ms.tensor(y)).numpy()
    expect = x @ y
    np.testing.assert_allclose(output, expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@pytest.mark.parametrize("shape1, shape2", [
    [[1280, 77, 64], [1280, 64, 77]],
    [[128, 256, 224], [128, 224, 256]],
])
@pytest.mark.parametrize("dtype", [
    np.float16, np.float32
])
def test_ops_network(context_mode, shape1, shape2, dtype):
    """
    Feature: BatchMatMulExt op.
    Description: test bmm_ext
    Expectation: expect correct shape result.
    """
    ms.set_context(mode=context_mode)

    bmm_cell = BmmCell()

    x = random_input(shape1, dtype)
    y = random_input(shape2, dtype)

    output = bmm_cell(ms.tensor(x), ms.tensor(y)).numpy()
    expect = x @ y
    allclose_nparray(output, expect, loss, loss)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_ops_dynamic():
    """
    Feature: ops.extend.add
    Description: dynamic shape and rank
    Expectation: success
    """

    x1 = ms.Tensor(random_input([10, 10, 10]))
    y1 = ms.Tensor(random_input([10, 10, 10]))
    x2 = ms.Tensor(random_input([20, 10, 10]))
    y2 = ms.Tensor(random_input([20, 10, 10]))

    TEST_OP(bmm, [[x1, y1], [x2, y2]], '', disable_input_check=True, disable_yaml_check=True)


def random_input(shape, dtype=np.float32):
    return np.random.randint(0, 10, shape).astype(dtype)
