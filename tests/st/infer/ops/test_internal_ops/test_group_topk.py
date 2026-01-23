# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
Tests for group topk operator.
"""
import numpy as np
import pytest

from mindspore import ops, nn
import mindspore as ms
from mindspore.common.np_dtype import bfloat16
from tests.mark_utils import arg_mark


def get_ms_dtype(np_dtype):
    if np_dtype == np.float32:
        ms_dtype = ms.float32
    elif np_dtype == np.float16:
        ms_dtype = ms.float16
    elif np_dtype == bfloat16:
        ms_dtype = ms.bfloat16
    else:
        ms_dtype = ms.float32  # default
    return ms_dtype


ms.set_context(device_target="Ascend",
               mode=ms.GRAPH_MODE,
               jit_config={"jit_level": "O0", "infer_boost": "on"},
               pynative_synchronize=True)

np.random.seed(42)


class GroupTopkCell(nn.Cell):
    def __init__(self):
        super().__init__()
        self.group_topk = ops.GroupTopk()
        self.dump = ops.TensorDump()

    def construct(self, token, idx_arr, group_num, k, k_inner):
        self.group_topk(token, idx_arr, group_num, k, k_inner)
        return token


def numpy_topk(arr, k, axis=-1):
    # 获取排序后的元素索引
    sorted_indices = np.argsort(arr, axis=axis)
    # 根据排序方向获取前 k 个元素的索引
    if axis < 0:
        axis = arr.ndim + axis
    topk_indices = np.take(sorted_indices, np.arange(-k, 0), axis=axis)
    # 根据索引获取前 k 个元素
    topk_values = np.take_along_axis(arr, topk_indices, axis=axis)
    return topk_values, topk_indices


def golden_np(inputx, token_num, expert_num, group_num, k, k_inner):
    input0 = inputx.reshape((token_num, group_num, expert_num // group_num))
    output = np.copy(input0)
    input0 = input0.astype(np.float32)
    group_tensor, _ = numpy_topk(input0, k_inner)
    group_tensor = np.sum(group_tensor, axis=-1)
    # The torch version of the CI is too old. Not support the stable parameter in torch.argsort.
    sort_index = np.argsort(-group_tensor, kind='stable')
    cols_to_use = np.arange(k, group_num, dtype=np.int64)
    row_indices = np.repeat(
        np.arange(sort_index.shape[0]), cols_to_use.shape[0])
    col_indices = sort_index[:, cols_to_use].reshape(-1)
    output[row_indices, col_indices] = 0
    return np.reshape(output, (token_num, expert_num))


def run(input_param, token_dtype, is_dynamic=False):
    token_num = input_param[0]
    expert_num = input_param[1]
    group_num = input_param[2]
    k = input_param[3]
    k_inner = input_param[4]
    input_shape = (token_num, expert_num)
    net = GroupTopkCell()
    if is_dynamic:
        input0_dyn = ms.Tensor(
            shape=[None] * len(input_shape), dtype=get_ms_dtype(token_dtype))
        input1_dyn = ms.Tensor(shape=[None], dtype=ms.int32)
        net.set_inputs(input0_dyn, input1_dyn, group_num, k, k_inner)

        for item in range(1, 6):
            input_shape = (token_num + item, expert_num)
            input0 = np.random.uniform(-2, 2, input_shape).astype(token_dtype)
            input1 = np.arange(1024, dtype=np.int32)
            input_tensor0 = ms.Tensor(input0, dtype=get_ms_dtype(token_dtype))
            # 用于复写 input_tensor0 = Parameter(input_tensor0)
            input_tensor1 = ms.Tensor(input1, dtype=ms.int32)
            ms_out = net(input_tensor0, input_tensor1, group_num, k, k_inner)
            golden_out = golden_np(input0, token_num + item, expert_num,
                                   group_num, k, k_inner)
            np.testing.assert_allclose(
                ms_out.astype(ms.float32).asnumpy(), golden_out.astype(np.float32), rtol=1e-2, atol=1e-2)

    else:
        input0 = np.random.uniform(-2, 2, input_shape).astype(token_dtype)
        input1 = np.arange(1024, dtype=np.int32)
        input_tensor0 = ms.Tensor(input0, dtype=get_ms_dtype(token_dtype))
        # 用于复写 input_tensor0 = Parameter(input_tensor0)
        input_tensor1 = ms.Tensor(input1, dtype=ms.int32)
        ms_out = net(input_tensor0, input_tensor1, group_num, k, k_inner)
        golden_out = golden_np(input0, token_num, expert_num,
                               group_num, k, k_inner)
        np.testing.assert_allclose(
            ms_out.astype(ms.float32).asnumpy(), golden_out.astype(np.float32), rtol=1e-2, atol=1e-2)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=104.00)
@pytest.mark.parametrize('token_dtype', [np.float16, bfloat16])
@pytest.mark.parametrize('input_param', [[64, 1024, 8, 3, 2], [359, 256, 8, 4, 2]])
@pytest.mark.parametrize('is_dynamic', [False, True])
def test_group_topk_float16(token_dtype, input_param, is_dynamic):
    """
    Feature: grouptopk st
    Description: input_param [tokenNum, 专家数, 分组数量, 选择前k个得分最大的组, 每组内前k_inner最大值求和]
    Expectation: success
    """
    run(input_param, token_dtype, is_dynamic)
