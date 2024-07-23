# Copyright 2023 Huawei Technologies Co., Ltd
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
import pytest
from tests.st.utils import test_utils

import mindspore as ms
from mindspore import ops
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def shape_forward_func(x):
    return ops.operations.manually_defined.shape_(x)

@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_shape_forward(mode):
    """
    Feature: Ops.
    Description: Test op shape forward.
    Expectation: Correct result.
    """
    ms.context.set_context(mode=mode)
    expect_out = (2, 2)
    x = ms.Tensor([[0, 1], [2, 3]], ms.float32)
    out = shape_forward_func(x)
    assert expect_out == out

@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_shape_dynamic(mode):
    """
    Feature: test dynamic tensor and dynamic scalar of shape.
    Description: test dynamic tensor and dynamic scalar of shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn1 = ms.Tensor(shape=None, dtype=ms.float32)
    expect_out1 = (1,)
    x1 = ms.Tensor([0], ms.float32)
    test_cell = test_utils.to_cell_obj(ops.operations.manually_defined.shape_)
    test_cell.set_inputs(x_dyn1)
    out1 = test_cell(x1)
    assert expect_out1 == out1
    x_dyn2 = ms.Tensor(shape=[None, None], dtype=ms.float32)
    expect_out2 = (2, 2)
    x2 = ms.Tensor([[0, 1], [2, 3]], ms.float32)
    test_cell.set_inputs(x_dyn2)
    out2 = test_cell(x2)
    assert expect_out2 == out2
