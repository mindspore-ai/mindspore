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

import mindspore
from mindspore.nn import Cell
from mindspore.common.generator import Generator

rtol = 1e-3


class UniformExtCell(Cell):
    def __init__(self):
        super().__init__()
        self.uniform = mindspore.ops.uniform_ext

    def construct(self, x, from_, to, generator):
        return self.uniform(x, from_, to, generator)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [
    mindspore.PYNATIVE_MODE,
    mindspore.GRAPH_MODE
])
def test_basic(context_mode):
    """
    Feature: extend.uniform
    Description: extend.uniform
    Expectation: Success
    """
    mindspore.set_context(jit_level='O0')
    mindspore.set_context(mode=context_mode)

    uniform_cell = UniformExtCell()

    x = random_input([64])
    from_ = 90.0
    to = 100.0

    g1 = Generator()
    g1.manual_seed(41)

    g2 = Generator()
    g2.manual_seed(41)

    output1 = uniform_cell(mindspore.tensor(x), from_, to, g1).numpy()
    output2 = uniform_cell(mindspore.tensor(x), from_, to, g1).numpy()
    expect1 = uniform_cell(mindspore.tensor(x), from_, to, g2).numpy()
    expect2 = uniform_cell(mindspore.tensor(x), from_, to, g2).numpy()
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


def random_input(shape):
    return np.random.rand(*shape).astype(np.float32)
