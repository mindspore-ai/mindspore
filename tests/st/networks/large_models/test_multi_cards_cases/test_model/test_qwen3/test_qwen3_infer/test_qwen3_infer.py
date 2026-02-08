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
"""Test Mcore Qwen3 inference"""
import os
import sys
import random
from pathlib import Path

import pytest

workspace = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    )))
sys.path.insert(0, os.path.join(workspace, "mindformers"))
from mindformers.tools.logger import logger

from tests.st.networks.large_models.test_multi_cards_cases.test_run_multi_cards_cases import TaskType

_LEVEL_0_TASK_TIME = 300
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.TWO_CARDS_TASK


class TestMcoreQwen3ParallelInference:
    """Test class for Qwen3 in inference"""

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_qwen3.py"
        assert self.run_script_path.exists(), f"Run script not found: {self.run_script_path}"

    @pytest.mark.level0
    def test_two_cards_cases(self):
        """Test two cards for Qwen3."""
        port_id = int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
        cmd_list = [
            "msrun",
            "--worker_num=2",
            "--local_worker_num=2",  # Should match NPU cards available
            f"--master_port={port_id}",  # Ensure port is unique per test run if parallelized at pytest level
            "--log_dir=./msrun_log_qwen3",
            "--join=True"]
        cmd_list += [
            str(self.run_script_path),
            "--device_num=2"
        ]
        cmd = " ".join(cmd_list)
        logger.info(f"Running command: {cmd}")
        return_code = os.system(cmd)
        assert return_code == 0, "Qwen3 inference st failed."
