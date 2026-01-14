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
from mindspore.ops.auto_generate import KVCacheScatterUpdate
from mindspore.common.parameter import Parameter
from mindspore import context, Tensor, jit
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark

kv_cache_scatter_update_op = KVCacheScatterUpdate()


@test_utils.run_with_cell
def kvcachescatterupdate_forward_func(var, indices, updates, axis, reduce):
    return kv_cache_scatter_update_op(var, indices, updates, axis, reduce)


def expect_func(var, indices, updates, axis):
    batch_size = var.shape[0]
    shape_1 = var.shape[1]
    if axis == -2:
        shape_not_axis = var.shape[3]
    elif axis == -1:
        shape_not_axis = var.shape[2]
    var_value = var
    indices_value = indices
    update_value = updates
    output = var_value
    if axis == -2:
        for i in range(batch_size):
            indices_key = indices_value[i]
            for o in range(shape_1):
                for k in range(shape_not_axis):
                    output[i][o][indices_key][k] = update_value[i][o][0][k]
    elif axis == -1:
        for i in range(batch_size):
            indices_key = indices_value[i]
            for o in range(shape_1):
                for k in range(shape_not_axis):
                    output[i][o][k][indices_key] = update_value[i][o][k][0]
    return output


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=20.75)
def test_kvcachescatterupdate_forward_mode():
    """
    Feature: Test kv_cache_scatter_update with static shape in GE.
    Description: call kv_cache_scatter_update with valid input.
    Expectation: return the correct value.
    """

    var_shape = [1, 5, 128, 4096]
    var = np.random.uniform(low=1, high=10, size=var_shape).astype(np.float32)
    indices_shape = [1]
    indices = np.random.randint(low=1, high=10, size=indices_shape).astype(np.int64)
    updates_shape = [1, 5, 128, 1]
    updates = np.random.uniform(low=1, high=10, size=updates_shape).astype(np.float32)

    output = (jit(kvcachescatterupdate_forward_func, backend="GE"))(Parameter(Tensor(var), "var"),
                                                                    Tensor(indices), Tensor(updates), -1, 'update')
    expect_shape = [1, 5, 128, 4096]
    assert np.allclose(output.shape, expect_shape)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=30.02)
@pytest.mark.parametrize('mode', ['GE', 'KBK', 'pynative'])
def test_scatter_value(mode):
    """
    Feature: Test kv_cache_scatter_update with static shape in GE.
    Description: call kv_cache_scatter_update with valid input.
    Expectation: return the correct value.
    """

    var_shape = [1, 5, 128, 4096]
    var = np.random.uniform(low=1, high=10, size=var_shape).astype(np.float32)
    indices_shape = [1]
    indices = np.random.randint(low=1, high=10, size=indices_shape).astype(np.int64)
    updates_shape = [1, 5, 128, 1]
    updates = np.random.uniform(low=1, high=10, size=updates_shape).astype(np.float32)

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        output = kvcachescatterupdate_forward_func(Tensor(var), Tensor(indices), Tensor(updates), -1, 'update')
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        output = (jit(kvcachescatterupdate_forward_func, jit_level="O0"))(Parameter(Tensor(var), "var"),
                                                                          Tensor(indices), Tensor(updates),
                                                                          -1, 'update')
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        output = (jit(kvcachescatterupdate_forward_func, backend="GE"))(Parameter(Tensor(var), "var"),
                                                                        Tensor(indices), Tensor(updates), -1, 'update')
    expect_value = expect_func(var, indices, updates, -1)
    assert np.allclose(output.asnumpy(), expect_value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
def test_ops_dynamic():
    """
    Feature: test kv_cache_scatter_update
    Description: dynamic shape and rank
    Expectation: success
    """
    var_shape = [1, 5, 128, 4096]
    indices_shape = [1]
    updates_shape = [1, 5, 128, 1]
    var_1 = Parameter(Tensor(np.random.uniform(low=1, high=10, size=var_shape).astype(np.float32)), "var")
    var_2 = Parameter(Tensor(np.random.uniform(low=1, high=10, size=var_shape).astype(np.float32)), "var")
    indices_1 = Tensor(np.random.randint(low=1, high=10, size=indices_shape).astype(np.int64))
    indices_2 = Tensor(np.random.randint(low=1, high=10, size=indices_shape).astype(np.int64))
    updates_1 = Tensor(np.random.uniform(low=1, high=10, size=updates_shape).astype(np.float32))
    updates_2 = Tensor(np.random.uniform(low=1, high=10, size=updates_shape).astype(np.float32))

    TEST_OP(kv_cache_scatter_update_op,
            [[var_1, indices_1, updates_1, -1, 'update'], [var_2, indices_2, updates_2, -1, 'update']],
            disable_case=['DiscontiguousInput', 'EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_grad': True,
                         'deterministic_use_origin_inputs': True})
