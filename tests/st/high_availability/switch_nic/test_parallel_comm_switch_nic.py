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
Driver script that wraps multi-card CommSwitchNic test cases with arg_mark
and launches them via msrun so MindSpore gatekeeper can execute them.
"""
import os
import random
import subprocess
from tests.mark_utils import arg_mark


BASE_PATH = os.getenv("TEST_BASE_PATH", "./")
WORKER_NUM = int(os.getenv("TEST_WORKER_NUM", "8"))
LOSS_FILE_NAME = os.getenv("TEST_LOSS_FILE_NAME", "loss_values.txt")
SUCCESS_MESSAGE = "==================== 1 passed in 20minutes ======================"


def setup_log_directory(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)


def run_msrun_command(test_case: str, log_dir: str, worker_num: int = 8, timeout: int = 600) -> int:
    port_hash = random.randint(10000, 60000)
    env = os.environ.copy()
    env.update({
        "HCCL_DETERMINISTIC": "true",
        "ASCEND_LAUNCH_BLOCKING": "1",
        "HCCL_NPU_SOCKET_PORT_RANGE": "16866-16888",
        "HCCL_HOST_SOCKET_PORT_RANGE": "50000-50031",
    })
    env.pop('HCCL_IF_BASE_PORT', None)
    command = [
        'msrun',
        f'--worker_num={worker_num}',
        '--join=True',
        f'--master_port={port_hash}',
        f'--log_dir={log_dir}',
        'pytest',
        '--disable-warnings',
        f'{BASE_PATH}/parallel_comm_switch_nic.py::{test_case}',
        '-s',
        '-v'
    ]
    result = subprocess.run(
        command,
        env=env,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False
    )
    if result.returncode != 0:
        error_msg = f"msrun execution failed for {test_case} (return code: {result.returncode})"
        if result.stderr:
            error_lines = result.stderr.split('\n')
            for line in error_lines[-20:]:
                if 'Error' in line or 'ERROR' in line or 'Traceback' in line:
                    error_msg += f"\n{line}"
        raise AssertionError(error_msg)
    if "1 passed" not in result.stdout:
        raise AssertionError(f"Test {test_case} did not pass")
    return result.returncode


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards', essential_mark='unessential')
def test_parallel_comm_switch_nic_01():
    '''
    Feature: Fabric Switchback (Single-Node)
    Description:
    Enable feature switch via `export MS_ENABLE_TFT="{TSP:1}"`. At step 10/12, \
    all cards execute _comm_switch_nic. Continue training after successful \
    switchback. Compare single/multi-device accuracy.
    Expectation:
    - No runtime errors.
    - Normal loss values in accuracy comparison.
    '''
    os.environ['MS_ENABLE_TFT'] = "{TSP:1}"
    test_case = "test_parallel_comm_switch_nic_01"
    log_dir = f"{BASE_PATH}/{test_case}_log"
    setup_log_directory(log_dir)
    run_msrun_command(test_case, log_dir)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_parallel_comm_switch_nic_02():
    '''
    Feature: Fabric Switchback (Single-Node)
    Description:
    Initiate model training. During step 10, all cards execute fabric switchback \
    via `_comm_switch_nic` with communication continuity check. Repeat operation \
    at step 12. Continue training to completion.
    Expectation:
    - Successful switchback at steps 10/12 (0 errors).
    - Seamless training resumption.
    - Final accuracy matches baseline.
    '''
    os.environ['MS_ENABLE_TFT'] = "{TSP:1}"
    test_case = "test_parallel_comm_switch_nic_02"
    log_dir = f"{BASE_PATH}/{test_case}_log"
    setup_log_directory(log_dir)
    run_msrun_command(test_case, log_dir)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_parallel_comm_switch_nic_03():
    '''
    Feature: Fabric Switchback (Single-Node Specific Devices)
    Description:
    Enable feature via `export MS_ENABLE_TFT="{TSP:1}"`. At step 11/12 (user input: step10/11), \
    devices 0 and 7 execute _comm_switch_nic. Continue training after successful \
    switchback. Compare single/multi-device accuracy.
    Expectation:
    - No runtime errors.
    - Normal loss values in accuracy comparison.
    '''
    os.environ['MS_ENABLE_TFT'] = "{TSP:1}"
    test_case = "test_parallel_comm_switch_nic_03"
    log_dir = f"{BASE_PATH}/{test_case}_log"
    setup_log_directory(log_dir)
    run_msrun_command(test_case, log_dir)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_parallel_comm_switch_nic_04():
    '''
    Feature: Fabric Switchback (Single-Node Specific Devices)
    Description:
    Enable feature via `export MS_ENABLE_TFT="{TSP:1}"`. At step 11/12 (user input: step10/11), \
    devices 0, 6 and 7 execute _comm_switch_nic. Continue training after switchback. \
    Compare single/multi-device accuracy.
    Expectation:
    - Switchback fails but training continues without impact.
    - No runtime errors.
    - Normal loss values in accuracy comparison.
    '''
    os.environ['MS_ENABLE_TFT'] = "{TSP:1}"
    test_case = "test_parallel_comm_switch_nic_04"
    log_dir = f"{BASE_PATH}/{test_case}_log"
    setup_log_directory(log_dir)
    run_msrun_command(test_case, log_dir)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_parallel_comm_switch_nic_05():
    '''
    Feature: Fabric Switchback (Checkpoint Step Operation)
    Description:
    Activate feature via `export MS_ENABLE_TFT="{TSP:1}"`. At weight-saving \
    step 10, rank 4 triggers _comm_switch_nic. Continue training after \
    switchback. Compare single/multi-device accuracy.
    Expectation:
    - Zero runtime errors during execution.
    - Normal loss values in accuracy comparison.
    '''
    os.environ['MS_ENABLE_TFT'] = "{TSP:1}"
    test_case = "test_parallel_comm_switch_nic_05"
    log_dir = f"{BASE_PATH}/{test_case}_log"
    setup_log_directory(log_dir)
    run_msrun_command(test_case, log_dir)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
