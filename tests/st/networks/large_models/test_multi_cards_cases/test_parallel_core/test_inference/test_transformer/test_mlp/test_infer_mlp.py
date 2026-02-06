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
"""mcore MLP UT of inference"""
import os
import sys
import random
import pytest
from pathlib import Path
import subprocess
import numpy as np

workspace = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    )))
sys.path.insert(0, os.path.join(workspace, "mindformers"))

from mindformers.tools.logger import logger

from tests.st.networks.large_models.test_multi_cards_cases.test_run_multi_cards_cases import TaskType
from tests.st.networks.large_models.test_multi_cards_cases.utils.double_benchmark import DoubleBenchmarkStandard, DoubleBenchmarkComparator
from tests.st.networks.large_models.test_multi_cards_cases.test_parallel_core.test_inference. \
test_transformer.test_mlp.data_gen_utils import (
    get_init_params,
    INPUT_SIZE,
    FFN_HIDDEN_SIZE,
    GOLDEN_DATA,
    GPU_DATA,
)

_LEVEL_0_TASK_TIME = 300
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.TWO_CARDS_TASK

TWO_CARD_TEST_PARAM = "model_args, data_keys, expect_error, tensor_parallel"
TWO_CARD_TEST_CASES = [
    (
        # 并行策略: 双卡tp=2, has_bias: FALSE, gated_linear_unit: TRUE, is_expert: FALSE, input_size: 32, ffn_hidden_size: 32
        # expected result: 功能跑通, 精度对齐。
        {"has_bias": False, "gated_linear_unit": True, "is_expert": False,
         "input_size": INPUT_SIZE, "ffn_hidden_size": FFN_HIDDEN_SIZE},
        {"output": "has_gate_output"},
        False,
        2
    ),
    (
        # 并行策略: 双卡tp=2, has_bias: True, gated_linear_unit: True, is_expert: FALSE, input_size: 32, ffn_hidden_size: 32
        # expected result: 功能跑通, 精度对齐。
        {"has_bias": True, "gated_linear_unit": True, "is_expert": False,
         "input_size": INPUT_SIZE, "ffn_hidden_size": FFN_HIDDEN_SIZE},
        {"output": "has_bias_gate_output"},
        False,
        2
    ),
]


def build_msrun_command_list(
        worker_num, local_worker_num, log_dir, run_script_path,
        input_size, ffn_hidden_size, has_bias,
        gated_linear_unit, output_path_param, tensor_parallel, port
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
        f"--ffn_hidden_size={ffn_hidden_size}",
        f"--has_bias={str(has_bias).lower()}",
        f"--gated_linear_unit={str(gated_linear_unit).lower()}",
        f"--output_path={output_path_param}",
        f"--tensor_parallel={tensor_parallel}"
    ]
    if input_size is not None:
        cmd_list.append(f"--input_size={input_size}")
    logger.info(f"Equivalent shell command for debugging (approximate): {' '.join(cmd_list)}")
    return cmd_list


class TestInferMLP:
    """Test class for MLP with different configurations"""
    OUTPUT_MS_FILENAME = "output_ms.npz"
    LOG_DIR_NAME = "msrun_log"
    WORKER_LOG_FILENAME = "worker_0.log"

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_infer_mlp.py"

    @staticmethod
    def check_function(output_ms_dict, model_args, data_keys):
        """
        Compare the shapes of output_ms and input_ms whether they are the same.
        """
        for key, _ in data_keys.items():
            output_data = output_ms_dict.get(key)
            input_data = get_init_params(
                model_args["input_size"] if model_args["input_size"] else INPUT_SIZE,
                model_args["ffn_hidden_size"])["input"]

            assert output_data.shape == input_data.shape, \
                (f"The shapes of output data and input data are different, "
                 f"got output shape: {output_data.shape} and input shape: {input_data.shape}")

    @staticmethod
    def check_acc(output_ms_dict, data_keys):
        """Compare output_ms with GOLDEN_DATA and GPU_DATA using DoubleBenchmarkComparator."""
        standard = DoubleBenchmarkStandard(dtype="bfloat16")

        for key, data_key in data_keys.items():
            npu_data = output_ms_dict.get(key)
            golden_data = GOLDEN_DATA.get(data_key)
            gpu_data = GPU_DATA.get(data_key)

            DoubleBenchmarkComparator.check_pass_or_not(
                npu_data=npu_data,
                gpu_data=gpu_data,
                golden_data=golden_data,
                standard=standard
            )

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
            expect_error=False,
            port=8118
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
            input_size=model_args["input_size"],
            ffn_hidden_size=model_args["ffn_hidden_size"],
            has_bias=model_args["has_bias"],
            gated_linear_unit=model_args["gated_linear_unit"],
            output_path_param=output_file_path,
            tensor_parallel=tensor_parallel,
            port=port
        )

        cmd_result = subprocess.run(
            cmd_list, shell=False, capture_output=True, text=True, check=False)

        if worker_num > 1:
            assert worker_log_file.exists()

        self.check_result(output_file_path, model_args, data_keys, cmd_result, expect_error)

    @pytest.mark.parametrize(TWO_CARD_TEST_PARAM, TWO_CARD_TEST_CASES)
    @pytest.mark.level0
    def test_two_cards_cases(self, model_args, data_keys, expect_error, tensor_parallel, tmp_path):
        """Test four cards cases with various configurations."""
        self.run_test(
            worker_num=tensor_parallel, local_worker_num=tensor_parallel,
            model_args=model_args, expect_error=expect_error,
            data_keys=data_keys,
            tmp_path=tmp_path,
            tensor_parallel=tensor_parallel,
            port=int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
        )
