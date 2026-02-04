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
import numpy as np
import pytest

from mindspore.ops.auto_generate import MoeComputeExpertTokens
from mindspore import Tensor, jit, JitConfig
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark



def get_expected_output(sorted_experts, num_expert):
    res = np.arange(num_expert)
    for i in range(num_expert):
        res[i] = np.searchsorted(sorted_experts, i, side='right')
    return res

@test_utils.run_with_cell
def moe_compute_expert_tokens(sorted_experts, num_expert):
    net = MoeComputeExpertTokens()
    return net(sorted_experts, num_expert)



@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=36.0)
@pytest.mark.parametrize('mode', ['GE', 'KBK', 'pynative'])
def test_moe_compute_expert_tokens_case0(mode):
    """
    Feature: Test the moe_compute_expert_tokens calculate
    Description: Test the moe_compute_expert_tokens ops in Ascend backend
    Expectation: The result match to the expect value.
    """
    sorted_expert_len = 302
    num_expert = 31
    random_int_list = []
    for _ in range(sorted_expert_len):
        random_int_list.append(np.random.randint(0, num_expert))
    sorted_experts = np.sort(random_int_list).astype(np.int32)

    if mode == 'pynative':
        expert_tokens = moe_compute_expert_tokens(Tensor(sorted_experts), num_expert)
    elif mode == 'KBK':
        expert_tokens = (jit(moe_compute_expert_tokens, jit_level="O0"))(Tensor(sorted_experts), num_expert)
    else:
        expert_tokens = (jit(moe_compute_expert_tokens, backend="GE"))(Tensor(sorted_experts), num_expert)
    expect_output = get_expected_output(sorted_experts, num_expert)

    np.testing.assert_allclose(expert_tokens.asnumpy(), expect_output)
