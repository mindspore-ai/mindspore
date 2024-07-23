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
from mindspore.ops.auto_generate import KVCacheScatterUpdate
from mindspore import Tensor, jit, JitConfig
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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

    output = (jit(kvcachescatterupdate_forward_func,
                  jit_config=JitConfig(jit_level="O2")))(Tensor(var), Tensor(indices), Tensor(updates), -1, 'update')
    expect_shape = [1, 5, 128, 4096]
    assert np.allclose(output.shape, expect_shape)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scatter_value():
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

    output = (jit(kvcachescatterupdate_forward_func,
                  jit_config=JitConfig(jit_level="O2")))(Tensor(var), Tensor(indices), Tensor(updates), -1, 'update')
    expect_value = expect_func(var, indices, updates, -1)
    assert np.allclose(output.asnumpy(), expect_value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ops_dynamic():
    """
    Feature: test kv_cache_scatter_update
    Description: dynamic shape and rank
    Expectation: success
    """
    var_shape = [1, 5, 128, 4096]
    indices_shape = [1]
    updates_shape = [1, 5, 128, 1]
    var_1 = Tensor(np.random.uniform(low=1, high=10, size=var_shape).astype(np.float32))
    var_2 = Tensor(np.random.uniform(low=1, high=10, size=var_shape).astype(np.float32))
    indices_1 = Tensor(np.random.randint(low=1, high=10, size=indices_shape).astype(np.int64))
    indices_2 = Tensor(np.random.randint(low=1, high=10, size=indices_shape).astype(np.int64))
    updates_1 = Tensor(np.random.uniform(low=1, high=10, size=updates_shape).astype(np.float32))
    updates_2 = Tensor(np.random.uniform(low=1, high=10, size=updates_shape).astype(np.float32))

    TEST_OP(kv_cache_scatter_update_op,
            [[var_1, indices_1, updates_1, -1, 'update'], [var_2, indices_2, updates_2, -1, 'update']],
            'kv_cache_scatter_update', disable_grad=True, disable_input_check=True)
