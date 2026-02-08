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
"""mcore MoE tp parallel UT of inference"""
import os
import sys
import random
from pathlib import Path
import subprocess
import pytest
import numpy as np

workspace = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
    )))
sys.path.insert(0, os.path.join(workspace, "mindformers"))

from mindformers.tools.logger import logger

from tests.st.networks.large_models.test_multi_cards_cases.test_parallel_core.test_inference. \
test_transformer.test_moe.test_moe_layer.data_gen_utils import (
    get_init_params,
    GOLDEN_DATA
)
from tests.st.networks.large_models.test_multi_cards_cases.utils.precision_checker import PrecisionChecker
from tests.st.networks.large_models.test_multi_cards_cases.test_run_multi_cards_cases import TaskType

_LEVEL_0_TASK_TIME = 300
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.FOUR_CARDS_TASK

MOE_CONFIG_WITH_SHARED_EXPERTS = {
    "batch_size": 4, "seq_len": 4, "hidden_size": 32, "moe_intermediate_size": 8, "num_experts": 8,
    "moe_shared_expert_intermediate_size": 8, "n_shared_experts": 1, "routed_scaling_factor": 2.5,
    "num_experts_per_tok": 2, "n_group": 2, "topk_group": 2
}

MOE_CONFIG_WITHOUT_SHARED_EXPERTS = {
    "batch_size": 4, "seq_len": 4, "hidden_size": 32, "moe_intermediate_size": 8, "num_experts": 8,
    "moe_shared_expert_intermediate_size": None, "n_shared_experts": 0, "routed_scaling_factor": 2.5,
    "num_experts_per_tok": 2, "n_group": 2, "topk_group": 2
}

SINGLE_CARD_TEST_PARAM = "model_args, data_keys, expect_error"
SINGLE_CARD_TEST_CASES = [
    (
        # 并行策略: 单卡
        # 配置：seq_len: 4, batch_size: 4, hidden_size: 32, num_experts: 8,
        # moe_intermediate_size: 8, moe_shared_expert_intermediate_size: 8,
        # n_shared_experts: 1, routed_scaling_factor: 2.5, num_experts_per_tok: 2,
        # n_group: 2, topk_group: 2
        # expected result: 功能跑通, 精度对齐。
        {"seq_len": 4, "batch_size": 4, "hidden_size": 32,
         "moe_intermediate_size": 8, "num_experts": 8, "moe_shared_expert_intermediate_size": 8,
         "n_shared_experts": 1, "routed_scaling_factor": 2.5, "num_experts_per_tok": 2,
         "n_group": 2, "topk_group": 2},
        {"output": "tp1"},
        False,
    ),
    (
        # 并行策略: 单卡
        # 配置：seq_len: 4, batch_size: 4, hidden_size: 32, num_experts: 8,
        # moe_intermediate_size: 8, moe_shared_expert_intermediate_size: None,
        # n_shared_experts: 0, routed_scaling_factor: 2.5, num_experts_per_tok: 2,
        # n_group: 2, topk_group: 2
        # expected result: 功能跑通, 精度对齐。
        {"seq_len": 4, "batch_size": 4, "hidden_size": 32,
         "moe_intermediate_size": 8, "num_experts": 8, "moe_shared_expert_intermediate_size": None,
         "n_shared_experts": 0, "routed_scaling_factor": 2.5, "num_experts_per_tok": 2,
         "n_group": 2, "topk_group": 2},
        {"output": "tp1_no_shared"},
        False,
    ),
]

FOUR_CARD_TEST_PARAM = "model_args, data_keys, expect_error, tensor_parallel, expert_parallel"
FOUR_CARD_DP1TP4EP1_TEST_CASES = [
    (
        # 并行策略: DP=1 TP=4 EP=1
        # eg: input (4 * 4, H) -> [moe] -> output (4 * 4,H)
        # seq_len: 4, batch_size: 4, hidden_size: 32, num_experts: 8,
        # moe_intermediate_size: 8, moe_shared_expert_intermediate_size: 8
        # expected result: 功能跑通, 精度对齐。
        MOE_CONFIG_WITH_SHARED_EXPERTS,
        {"output": "tp1"},
        False,
        4, 1),
    (
        # 并行策略: DP=1 TP=4 EP=1
        # eg: input (4 * 4, H) -> [moe] -> output (4 * 4,H)
        # seq_len: 4, batch_size: 4, hidden_size: 32, num_experts: 8,
        # moe_intermediate_size: 8, moe_shared_expert_intermediate_size: None
        # expected result: 功能跑通, 精度对齐。
        MOE_CONFIG_WITHOUT_SHARED_EXPERTS,
        {"output": "tp1_no_shared"},
        False,
        4, 1)
]


def build_msrun_command_list(
        worker_num, local_worker_num, log_dir, run_script_path, seq_len, batch_size,
        num_experts, hidden_size, moe_intermediate_size, n_shared_experts, routed_scaling_factor,
        num_experts_per_tok, n_group, topk_group, moe_shared_expert_intermediate_size,
        output_path_param, tensor_parallel, expert_parallel=1, port=8118
):
    """ Build the msrun command with the specified parameters. """
    if worker_num == 1:
        cmd_list = ["python"]
    else:
        cmd_list = [
            "msrun",
            f"--worker_num={worker_num}",
            f"--local_worker_num={local_worker_num}",  # Should match NPU cards available
            f"--master_port={port}", # Ensure port is unique per test run if parallelized at pytest level
            f"--log_dir={log_dir}",
            "--join=True"]
    cmd_list += [
        str(run_script_path),
        f"--seq_len={seq_len}",
        f"--batch_size={batch_size}",
        f"--num_experts={num_experts}",
        f"--hidden_size={hidden_size}",
        f"--moe_intermediate_size={moe_intermediate_size}",
        f"--n_shared_experts={n_shared_experts}",
        f"--routed_scaling_factor={routed_scaling_factor}",
        f"--num_experts_per_tok={num_experts_per_tok}",
        f"--n_group={n_group}",
        f"--topk_group={topk_group}",
        f"--output_path={output_path_param}",
        f"--tensor_parallel={tensor_parallel}",
        f"--expert_parallel={expert_parallel}"
    ]
    if moe_shared_expert_intermediate_size is not None:
        cmd_list.append(f"--moe_shared_expert_intermediate_size={moe_shared_expert_intermediate_size}")
    logger.info(f"Equivalent shell command for debugging (approximate): {' '.join(cmd_list)}")
    return cmd_list


class TestInferMoE:
    """Test class for MLP with different configurations"""
    OUTPUT_MS_FILENAME = "output_ms.npz"
    LOG_DIR_NAME = "msrun_log"
    WORKER_LOG_FILENAME = "worker_0.log"

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_infer_moe.py"

    @staticmethod
    def check_function(output_ms_dict, model_args, data_keys):
        """
        Compare the shapes of output_ms and input_ms whether they are the same.
        """
        for key, _ in data_keys.items():
            output_data = output_ms_dict.get(key)
            input_data = get_init_params(
                model_args["seq_len"], model_args["batch_size"], model_args["hidden_size"],
                model_args["num_experts"], model_args["moe_intermediate_size"])["input"]

            assert output_data.shape == input_data.shape, \
                (f"The shapes of output data and input data are different, "
                 f"got output shape: {output_data.shape} and input shape: {input_data.shape}")

    @staticmethod
    def check_acc(output_ms_dict, data_keys):
        """Compare output_ms with GOLDEN_DATA and GPU_DATA."""

        for key, data_key in data_keys.items():
            npu_data = output_ms_dict.get(key)
            golden_data = GOLDEN_DATA.get(data_key)
            res_npu_golden = np.allclose(npu_data, golden_data, rtol=0.004)

            assert res_npu_golden

    def check_result(
            self,
            output_file_path,
            model_args,
            data_keys,
            result,
            expect_error
    ):
        """Helper function to check results"""

        if expect_error:
            assert result.returncode != 0, (
                f"Expected an error but test script passed. "
                f"Stdout:\n{result.stdout}\n"
                f"Stderr:\n{result.stderr}"
            )
        else:
            assert result.returncode == 0, (
                f"Test script failed with non-zero exit code: "
                f"{result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
            )
            assert output_file_path.exists(), (
                f"Output file {output_file_path} was not created."
            )

            output_ms_dict = np.load(output_file_path)

            # check whether the function of rotary embedding module works properly.
            self.check_function(output_ms_dict, model_args, data_keys)
            # Check whether the accuracy is consistent with vLLM
            self.check_acc(output_ms_dict, data_keys)

    def run_test(
            self,
            worker_num,
            local_worker_num,
            model_args,
            data_keys,
            tmp_path,
            tensor_parallel=1,
            expert_parallel=1,
            expect_error=False,
            port=8118,
    ):
        """Helper function to run test and check results"""
        output_file_path = tmp_path / self.OUTPUT_MS_FILENAME
        log_dir_path = tmp_path / self.LOG_DIR_NAME
        log_dir_path.mkdir(parents=True, exist_ok=True)
        worker_log_file = log_dir_path / self.WORKER_LOG_FILENAME

        cmd_list = build_msrun_command_list(
            worker_num=worker_num,
            local_worker_num=local_worker_num,
            log_dir=log_dir_path,
            run_script_path=self.run_script_path,
            seq_len=model_args["seq_len"],
            batch_size=model_args["batch_size"],
            num_experts=model_args["num_experts"],
            hidden_size=model_args["hidden_size"],
            moe_intermediate_size=model_args["moe_intermediate_size"],
            moe_shared_expert_intermediate_size=model_args["moe_shared_expert_intermediate_size"],
            n_shared_experts=model_args["n_shared_experts"],
            routed_scaling_factor=model_args["routed_scaling_factor"],
            num_experts_per_tok=model_args["num_experts_per_tok"],
            n_group=model_args["n_group"],
            topk_group=model_args["topk_group"],
            output_path_param=output_file_path,
            tensor_parallel=tensor_parallel,
            expert_parallel=expert_parallel,
            port=port
        )

        cmd_result = subprocess.run(
            cmd_list, shell=False, capture_output=True, text=True, check=False)

        if worker_num > 1:
            assert worker_log_file.exists()

        self.check_result(output_file_path, model_args, data_keys, cmd_result, expect_error)

    @pytest.mark.parametrize(SINGLE_CARD_TEST_PARAM, SINGLE_CARD_TEST_CASES)
    @pytest.mark.level1
    def test_single_card_cases(self, model_args, data_keys, expect_error, tmp_path):
        """Test single card cases with various configurations."""
        self.run_test(
            worker_num=1, local_worker_num=1,
            model_args=model_args, expect_error=expect_error,
            data_keys=data_keys,
            tmp_path=tmp_path
        )


class TestInferMoELayerTP(TestInferMoE):
    """Test class for MoELayer with tp parallel"""

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_infer_moe_layer.py"

    @staticmethod
    def check_acc(output_ms_dict, data_keys):
        """Compare output_ms with GOLDEN_DATA and GPU_DATA."""

        checker = PrecisionChecker()
        for key, data_key in data_keys.items():
            npu_data = output_ms_dict.get(key)
            golden_data = GOLDEN_DATA.get(data_key)
            checker.check_precision(golden_data, npu_data)

    @pytest.mark.parametrize(FOUR_CARD_TEST_PARAM, FOUR_CARD_DP1TP4EP1_TEST_CASES)
    @pytest.mark.level0
    def test_four_cards_tp4ep1_cases(
            self, model_args, data_keys, expect_error,
            tensor_parallel, expert_parallel, tmp_path
    ):
        """Test four-card cases with dp1-tp4-ep1 parallel configurations."""
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
