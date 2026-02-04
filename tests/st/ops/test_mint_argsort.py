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

from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP

import mindspore as ms
import mindspore.mint as mint


@test_utils.run_with_cell
def argsort_forward(input_x, dim=-1, descending=False, stable=False):
    return mint.argsort(input_x, dim, descending, stable)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_argsort_normal(mode):
    """
    Feature: argsort
    Description: Verify the result of argsort.
    Expectation: success
    """
    ms.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={"jit_level": "O0"})
    a = [[0.0785, 1.5267, -0.8521, 0.4065],
         [0.1598, 0.0788, -0.0745, -1.2700],
         [1.2208, 1.0722, -0.7064, 1.2564],
         [0.0669, -0.2318, -0.8229, -0.9280]]
    x = ms.Tensor(a)
    out = argsort_forward(x)
    expect = [[2, 0, 3, 1],
              [3, 2, 1, 0],
              [2, 1, 0, 3],
              [3, 2, 1, 0]]
    assert np.allclose(out.asnumpy(), np.array(expect))


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend'], level_mark='level2', card_mark='onecard',
          essential_mark='essential')
def test_argsort_dynamic_shape():
    """
    Feature: mint.argsort
    Description: Verify the result of argsort forward with dynamic shape
    Expectation: success
    """

    inputs1 = ms.Tensor(np.array([[1, 10, 2], [0, 6, 1]], np.float32))
    inputs2 = ms.Tensor(np.array([[[5, 0.1, -1.2], [0, 5.5, 1.2]], [[-5, 0.1, 1.2], [0, -5.5, 1.2]]], np.float32))


    TEST_OP(argsort_forward, [[inputs1, 1, False, True], [inputs2, -1, True, False]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['ScalarTensor'],
            case_config={'disable_grad': True})
