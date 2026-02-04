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
"""mcore MoE dp parallel UT of inference"""
import os
import random

import pytest

from tests.st.networks.large_models.test_multi_cards_cases.test_parallel_core.test_inference.test_transformer. \
test_moe.test_moe_layer.test_infer_moe_tp import (
        MOE_CONFIG_WITH_SHARED_EXPERTS,
        FOUR_CARD_TEST_PARAM,
        TestInferMoELayerTP,
    )
from tests.st.networks.large_models.test_multi_cards_cases.test_run_multi_cards_cases import TaskType

_LEVEL_0_TASK_TIME = 300
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.FOUR_CARDS_TASK

FOUR_CARD_DP4TP1EP1_TEST_CASES = [
    (
        # 并行策略: DP=4 TP=1 EP=1
        # eg: input (4 * 4, H) -> 按卡切分 (1 * 4, H) -> [moe] (4 * 4, H) -> allgather -> output (4 * 4, H)
        # seq_len: 4, batch_size: 4, hidden_size: 32, num_experts: 8,
        # moe_intermediate_size: 8, moe_shared_expert_intermediate_size: 8
        # expected result: 功能跑通, 精度对齐。
        MOE_CONFIG_WITH_SHARED_EXPERTS,
        {"output": "tp1"},
        False,
        1, 1),
]


class TestInferMoELayerDPParallel(TestInferMoELayerTP):
    """Test class for InferMoELayer with dp parallel"""
    @pytest.mark.parametrize(FOUR_CARD_TEST_PARAM, FOUR_CARD_DP4TP1EP1_TEST_CASES)
    @pytest.mark.level0
    def test_four_cards_dp4_cases(
            self, model_args, data_keys, expect_error,
            tensor_parallel, expert_parallel, tmp_path
    ):
        """Test four-card cases with dp4-tp1-ep1 parallel configurations."""
        self.run_test(
            worker_num=4,
            local_worker_num=4,
            model_args=model_args,
            expect_error=expect_error,
            data_keys=data_keys,
            tensor_parallel=tensor_parallel,
            expert_parallel=expert_parallel,
            tmp_path=tmp_path,
            port=int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
        )
