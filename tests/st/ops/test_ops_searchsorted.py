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

import numpy as np
import pytest
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from mindspore import ops, nn
import mindspore as ms

class SearchSortedNet(nn.Cell):
    def __init__(self):
        super(SearchSortedNet, self).__init__()
        self.searchsorted = ops.searchsorted

    def construct(self, x1, x2, out_int32=False, right=False, side="left", sorter=None):
        return self.searchsorted(x1, x2, out_int32=out_int32, right=right, side=side, sorter=sorter)


def generate_random_input(shape, dtype):
    return np.sort(np.random.uniform(0.9, 1.0, size=shape).astype(dtype)), \
           np.random.uniform(0.9, 1.0, size=shape).astype(dtype)

def searchsorted_forward_func(x, y, out_int32=False, right=False, sorter=None):
    return SearchSortedNet()(x, y, out_int32=out_int32, right=right, side="right", sorter=sorter)

def generate_expect_forward_output(x, y, side="right", sorter=None):
    return np.searchsorted(x, y, side=side, sorter=sorter)

@test_utils.run_with_cell
def searchsorted_vmap_func(x, y, out_int32=False, right=False, sorter=None):
    return ops.vmap(searchsorted_forward_func, in_axes=(0, 0, None, None, None),
                    out_axes=0)(x, y, out_int32, right, sorter)

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_searchsorted_forward(context_mode):
    """
    Feature: Ops.
    Description: test op searchsorted forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_np = np.array([-1, 0.0, 1.0, 2.0], dtype=np.float32)
    y_np = np.array([0.0, 1.0, 2.0, -2], dtype=np.float32)
    x = ms.Tensor(x_np)
    y = ms.Tensor(y_np)
    out = searchsorted_forward_func(x, y)
    expect_out = generate_expect_forward_output(x_np, y_np)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_searchsorted_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function searchsorted vmap.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=context_mode)
    x_np = np.array([[-1, 0.0, 1.0, 2.0], [-1, 0.0, 1.0, 2.0]], dtype=np.float32)
    y_np = np.array([[0.0, 1.0, 2.0, -2], [0.0, 1.0, 2.0, -2]], dtype=np.float32)
    x = ms.Tensor(x_np)
    y = ms.Tensor(y_np)
    out = searchsorted_vmap_func(x, y)
    expect_out = []
    for s_x_np, s_y_np in zip(x_np, y_np):
        np_out = generate_expect_forward_output(s_x_np, s_y_np)
        expect_out.append(np_out)

    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_searchsorted_dynamic(context_mode):
    """
    Feature: pyboost function.
    Description: test function searchsorted dynamic.
    Expectation: expect correct result.
    """

    x1, _ = generate_random_input((20), np.float32)
    _, other1 = generate_random_input((2, 3, 4, 5), np.float32)
    out_int32_1 = False
    right_1 = False
    sorter_1 = ms.Tensor(np.argsort(x1, axis=-1))

    x2, _ = generate_random_input((2, 3, 4), np.float32)
    _, other2 = generate_random_input((2, 3, 4), np.float32)
    out_int32_2 = True
    right_2 = True
    sorter_2 = ms.Tensor(np.argsort(x2, axis=-1))

    TEST_OP(searchsorted_forward_func,
            [[ms.Tensor(x1), ms.Tensor(other1), out_int32_1, right_1, sorter_1],
             [ms.Tensor(x2), ms.Tensor(other2), out_int32_2, right_2, sorter_2]], 'searchsorted')
