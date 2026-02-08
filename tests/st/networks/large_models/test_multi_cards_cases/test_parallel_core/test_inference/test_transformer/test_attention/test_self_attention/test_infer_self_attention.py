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
"""mcore Attention UT of inference"""
import os
import sys
import random
import subprocess
from pathlib import Path
import numpy as np
import pytest

workspace = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
    )))
sys.path.insert(0, os.path.join(workspace, "mindformers"))

from mindformers.tools.logger import logger

from tests.st.networks.large_models.test_multi_cards_cases.test_run_multi_cards_cases import TaskType
from tests.st.networks.large_models.test_multi_cards_cases.utils.double_benchmark import DoubleBenchmarkStandard, DoubleBenchmarkComparator
from tests.st.networks.large_models.test_multi_cards_cases.test_parallel_core.test_inference. \
test_transformer.test_attention.test_self_attention.data_gen_utils import (
    get_init_params,
    BATCH_SIZE,
    PREFILL_SEQ_LEN,
    DECODE_SEQ_LEN,
    NUM_HEADS,
    HIDDEN_SIZE,
    GOLDEN_DATA,
    GPU_DATA
)


_LEVEL_0_TASK_TIME = 300
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.TWO_CARDS_TASK

TWO_CARD_TEST_PARAM = "model_args, data_keys, expect_error, tensor_parallel"
TWO_CARD_TEST_CASES = [
    (
        # 并行策略: 双卡, batch_size: 2, prefill_seq_len: 2, decode_seq_len: 1,
        # num_heads: 2, num_query_groups: 2, hidden_size: 32, use_flash_attention: TRUE
        # expected_result: 功能跑通。
        {"batch_size": BATCH_SIZE, "prefill_seq_len": PREFILL_SEQ_LEN,
         "decode_seq_len": DECODE_SEQ_LEN, "num_heads": NUM_HEADS, "num_query_groups": 2,
         "hidden_size": HIDDEN_SIZE, "use_flash_attention": True},
        {"prefill_output": "prefill_output_1", "decode_output": "decode_output_1"},
        False,
        2
    )
]


def build_msrun_command_list(
        worker_num, local_worker_num, log_dir, run_script_path,
        batch_size, prefill_seq_len, decode_seq_len, num_heads, num_query_groups,
        hidden_size, use_flash_attention, output_path_param, tensor_parallel, port
):
    """ Build the msrun command with the specified parameters. """
    if worker_num == 1:
        cmd_list = ["python"]
    else:
        cmd_list = [
            "msrun",
            f"--worker_num={worker_num}",
            f"--local_worker_num={local_worker_num}",
            f"--master_port={port}",
            f"--log_dir={log_dir}",
            "--join=True"
        ]
    cmd_list += [
        str(run_script_path),
        f"--batch_size={batch_size}",
        f"--prefill_seq_len={prefill_seq_len}",
        f"--decode_seq_len={decode_seq_len}",
        f"--num_heads={num_heads}",
        f"--num_query_groups={num_query_groups}",
        f"--hidden_size={hidden_size}",
        f"--use_flash_attention={str(use_flash_attention).lower()}",
        f"--output_path={output_path_param}",
        f"--tensor_parallel={tensor_parallel}"
    ]
    logger.info(f"Equivalent shell command for debugging (approximate): {' '.join(cmd_list)}")
    return cmd_list


class TestInferSelfAttention:
    """Test class for self_attn with different configurations"""
    OUTPUT_MS_FILENAME = "output_ms.npz"
    LOG_DIR_NAME = "msrun_log"
    WORKER_LOG_FILENAME = "worker_0.log"

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_infer_self_attention.py"

    @staticmethod
    def check_function(output_ms_dict, data_keys, n_kv_heads):
        """
        Compare the shapes of output_ms and input_ms whether they are the same.
        """
        for key, _ in data_keys.items():
            output_data = output_ms_dict.get(key)
            if key == "prefill_output":
                source_hidden_states = get_init_params(n_kv_heads=n_kv_heads)["prefill_hidden_states"]
                hidden_states = source_hidden_states.reshape(
                    BATCH_SIZE * PREFILL_SEQ_LEN, NUM_HEADS * int(HIDDEN_SIZE / NUM_HEADS)
                )
            else:
                source_hidden_states = get_init_params(n_kv_heads=n_kv_heads)["decoder_hidden_states"]
                hidden_states = source_hidden_states.reshape(
                    BATCH_SIZE * min(1, DECODE_SEQ_LEN),
                    NUM_HEADS * int(HIDDEN_SIZE / NUM_HEADS)
                )

            assert np.array_equal(output_data.shape, hidden_states.shape), \
                (f"The shapes of output data and input data are different, "
                 f"got output shape: {output_data.shape} and input shape: {hidden_states.shape}")

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

        cmd_list = build_msrun_command_list(
            worker_num=worker_num,
            local_worker_num=local_worker_num,
            log_dir=log_dir_path,
            run_script_path=self.run_script_path,
            batch_size=model_args["batch_size"],
            prefill_seq_len=model_args["prefill_seq_len"],
            decode_seq_len=model_args["decode_seq_len"],
            num_heads=model_args["num_heads"],
            num_query_groups=model_args["num_query_groups"],
            hidden_size=model_args["hidden_size"],
            use_flash_attention=model_args["use_flash_attention"],
            output_path_param=output_file_path,
            tensor_parallel=tensor_parallel,
            port=port
        )

        result = subprocess.run(
            cmd_list, shell=False, capture_output=True, text=True, check=False)

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

            # check whether the function of self_attn works properly.
            self.check_function(output_ms_dict, data_keys, model_args["num_query_groups"])

            self.check_acc(output_ms_dict, data_keys)

    @pytest.mark.parametrize(TWO_CARD_TEST_PARAM, TWO_CARD_TEST_CASES)
    @pytest.mark.level0
    def test_two_cards_configurations(self, model_args, data_keys, expect_error, tensor_parallel, tmp_path):
        """Test two cards with various configurations."""
        os.environ['MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST'] = 'PagedAttention'
        self.run_test(
            worker_num=tensor_parallel, local_worker_num=tensor_parallel,
            model_args=model_args, expect_error=expect_error,
            data_keys=data_keys,
            tmp_path=tmp_path,
            tensor_parallel=tensor_parallel,
            port=int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
        )
