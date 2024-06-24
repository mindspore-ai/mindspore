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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore
from mindspore import Tensor
from mindspore.ops.operations.random_ops import UniformExt
from mindspore.nn import Cell

from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP

rtol = 1e-3


class UniformExtCell(Cell):
    def __init__(self):
        super().__init__()
        self.uniform = UniformExt()

    def construct(self, x, from_, to, seed, offset):
        return self.uniform(x, from_, to, seed, offset)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [
    mindspore.PYNATIVE_MODE,
    mindspore.GRAPH_MODE
])
def test_basic(context_mode):
    """
    Feature: UniformExt
    Description: UniformExt
    Expectation: Success
    """
    mindspore.set_context(jit_level='O0')
    mindspore.set_context(mode=context_mode)

    uniform_cell = UniformExtCell()

    x = random_input([10, 10, 10])
    from_ = 90.0
    to = 100.0

    seed1 = Tensor(41, dtype=mindspore.int64)
    offset1 = Tensor(0, dtype=mindspore.int64)
    # seed1 and offset1 would be modified in pyboost, so a copy is needed
    seed1_2 = Tensor(41, dtype=mindspore.int64)
    offset1_2 = Tensor(0, dtype=mindspore.int64)

    seed2 = Tensor(42, dtype=mindspore.int64)
    offset2 = Tensor(3, dtype=mindspore.int64)
    seed2_2 = Tensor(42, dtype=mindspore.int64)
    offset2_2 = Tensor(3, dtype=mindspore.int64)

    output1 = uniform_cell(mindspore.tensor(x), from_, to, seed1, offset1).numpy()
    expect1 = uniform_cell(mindspore.tensor(x), from_,
                           to, seed1_2, offset1_2).numpy()
    output2 = uniform_cell(mindspore.tensor(x), from_, to, seed2, offset2).numpy()
    expect2 = uniform_cell(mindspore.tensor(x), from_,
                           to, seed2_2, offset2_2).numpy()
    np.testing.assert_allclose(output1, expect1, rtol=rtol)

    mean1 = output1.mean()
    std1 = output1.std()

    expect_mean1 = (from_ + to) / 2
    expect_std1 = (to - from_) / np.sqrt(12)

    assert np.isclose(mean1, expect_mean1, rtol=0.01)
    assert np.isclose(std1, expect_std1, rtol=0.1)

    mean2 = output2.mean()
    std2 = output2.std()

    expect_mean2 = (from_ + to) / 2
    expect_std2 = (to - from_) / np.sqrt(12)

    assert np.isclose(mean2, expect_mean2, rtol=0.01)
    assert np.isclose(std2, expect_std2, rtol=0.1)

    assert not np.allclose(output1, output2, rtol=rtol)
    assert not np.allclose(expect1, expect2, rtol=rtol)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_op():
    """
    Feature: TEST_OP
    Description: TEST_OP
    Expectation: Success
    """
    x1 = random_input((10, 10))
    x2 = random_input((10, 10, 10))

    from_ = 90.0
    to = 100.0

    seed1 = Tensor(41, dtype=mindspore.int64)
    offset1 = Tensor(0, dtype=mindspore.int64)

    seed2 = Tensor(42, dtype=mindspore.int64)
    offset2 = Tensor(3, dtype=mindspore.int64)

    TEST_OP(UniformExtCell(), [
        [mindspore.Tensor(x1), from_, to, seed1, offset1],
        [mindspore.Tensor(x2), from_, to, seed2, offset2],
    ], 'uniform_ext', disable_input_check=True, disable_mode=['GRAPH_MODE'], disable_grad=True, inplace_update=True)


def random_input(shape):
    return np.random.rand(*shape).astype(np.float32)
