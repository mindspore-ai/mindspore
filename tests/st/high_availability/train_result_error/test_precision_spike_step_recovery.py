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
""" test_precision_spike_step_recovery """
import os
import subprocess
from tests.mark_utils import arg_mark

# Constants
BASE_PATH = "./"
WORKER_NUM = 8
LOSS_FILE_NAME = "loss_values.txt"
SUCCESS_MESSAGE = "==================== 1 passed in 20minutes ======================"
COMMON_ENV_VARS = {
    "HCCL_DETERMINISTIC": "true",
    "ASCEND_LAUNCH_BLOCKING": "1",
}
ERROR_REGEX = "Traceback|Error|ERROR"
PASSED_PATTERN = "1 passed"


def setup_log_directory(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)


def setup_environment(custom_env: dict[str, str]) -> None:
    env_vars = {**COMMON_ENV_VARS, **custom_env}
    for key, value in env_vars.items():
        os.environ[key] = value


def run_msrun_command(test_case: str, log_dir: str) -> int:
    command = (
        f"msrun --worker_num={WORKER_NUM} --join=True --log_dir={log_dir} pytest --disable-warnings "
        f"{BASE_PATH}/precision_spike_step_recovery.py::{test_case} -s -v"
    )
    ret = os.system(command)
    if ret != 0:
        error_cmd = f"grep -rE '{ERROR_REGEX}' {log_dir}/worker_*"
        result = subprocess.run(error_cmd, shell=True, stdout=subprocess.PIPE, text=True, check=True)
        print(result.stdout)
    assert ret == 0, f"msrun execution failed with return code {ret} (return code:{ret//256})"
    return ret


def extract_loss_values(log_dir: str) -> list[str]:
    worker7_log = f"{log_dir}/worker_7.log"
    loss_file = f"{log_dir}/{LOSS_FILE_NAME}"
    grep_cmd = f"grep -oP 'loss: \\K\\d+\\.\\d+' {worker7_log} > {loss_file}"
    os.system(grep_cmd)
    with open(loss_file, "r", encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def verify_workers_passed(log_dir: str) -> None:
    passed_cmd = f"grep -roh '{PASSED_PATTERN}' {log_dir} | wc -l"
    result_proc = subprocess.run(
        passed_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
    )
    assert result_proc.returncode == 0, f"Failed to count passed workers: {result_proc.stderr}"
    passed_count = int(result_proc.stdout.strip())
    assert passed_count == WORKER_NUM, f"Workers pass count mismatch: {passed_count}/{WORKER_NUM}"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark="allcards", essential_mark='essential')
def test_precision_spike_step_recovery_01() -> None:
    '''
    Feature: precision spike step recovery
    Description: Test precision spike step recovery with TRE enabled
                 Verify loss consistency between step 6 and 7, and all workers pass
    Expectation: Loss at step 6 equals step 7, and 8 workers all pass
    '''
    test_case = "test_precision_spike_step_recovery_01"
    log_dir = f"{BASE_PATH}/{test_case}_log"
    setup_log_directory(log_dir)
    setup_environment({"MS_ENABLE_TFT": "{TRE:2,TRE_SNAPSHOT_STEPS:5}"})
    run_msrun_command(test_case, log_dir)
    loss_lines = extract_loss_values(log_dir)
    lineA = loss_lines[5]
    lineB = loss_lines[6]
    assert lineA == lineB, f"Loss inconsistency: step6={lineA}, step7={lineB}"
    verify_workers_passed(log_dir)
    print(SUCCESS_MESSAGE)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark="allcards",
        essential_mark='essential')
def test_precision_spike_step_recovery_02() -> None:
    '''
    Feature: precision spike step recovery
    Description: Test precision spike step recovery with TRE enabled (TRE_SNAPSHOT_STEPS=5)
                 Verify loss consistency between (step6&7), (step8&10), (step11&14), and all workers pass
    Expectation: Loss pairs (6=7, 8=10, 11=14) are equal, and 8 workers all pass
    '''
    test_case = "test_precision_spike_step_recovery_02"
    log_dir = f"{BASE_PATH}/{test_case}_log"
    setup_log_directory(log_dir)
    setup_environment({"MS_ENABLE_TFT": "{TRE:2,TRE_SNAPSHOT_STEPS:5}"})
    run_msrun_command(test_case, log_dir)
    loss_lines = extract_loss_values(log_dir)
    lineA1, lineA2 = loss_lines[5], loss_lines[6]
    lineB1, lineB2 = loss_lines[7], loss_lines[9]
    lineC1, lineC2 = loss_lines[10], loss_lines[13]
    assert lineA1 == lineA2, f"Loss inconsistency: step6={lineA1}, step7={lineA2}"
    assert lineB1 == lineB2, f"Loss inconsistency: step8={lineB1}, step10={lineB2}"
    assert lineC1 == lineC2, f"Loss inconsistency: step11={lineC1}, step14={lineC2}"
    verify_workers_passed(log_dir)
    print(SUCCESS_MESSAGE)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark="allcards", essential_mark='essential')
def test_precision_spike_step_recovery_pp() -> None:
    '''
    Feature: precision spike step recovery
    Description: Test precision spike step recovery with TRE enabled (TRE_SNAPSHOT_STEPS=1)
                 Verify loss consistency between (step3&4), (step5&6), (step7&8), and all workers pass
    Expectation: Loss pairs (3=4, 5=6, 7=8) are equal, and 8 workers all pass
    '''
    test_case = "test_precision_spike_step_recovery_03"
    log_dir = f"{BASE_PATH}/{test_case}_log"
    setup_log_directory(log_dir)
    setup_environment({"MS_ENABLE_TFT": "{TRE:2,TRE_SNAPSHOT_STEPS:1}"})
    run_msrun_command(test_case, log_dir)
    loss_lines = extract_loss_values(log_dir)
    lineA1, lineA2 = loss_lines[2], loss_lines[3]
    lineB1, lineB2 = loss_lines[4], loss_lines[5]
    lineC1, lineC2 = loss_lines[6], loss_lines[7]
    assert lineA1 == lineA2, f"Loss inconsistency: step3={lineA1}, step4={lineA2}"
    assert lineB1 == lineB2, f"Loss inconsistency: step5={lineB1}, step6={lineB2}"
    assert lineC1 == lineC2, f"Loss inconsistency: step7={lineC1}, step8={lineC2}"
    verify_workers_passed(log_dir)
    print(SUCCESS_MESSAGE)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark="allcards", essential_mark='essential')
def test_precision_spike_step_recovery_04() -> None:
    '''
    Feature: precision spike step recovery
    Description: Test precision spike step recovery with TRE enabled (TRE_SNAPSHOT_STEPS=5)
                 No loss consistency check, only verify all workers pass
    Expectation: 8 workers all pass
    '''
    test_case = "test_precision_spike_step_recovery_04"
    log_dir = f"{BASE_PATH}/{test_case}_log"
    setup_log_directory(log_dir)
    setup_environment({"MS_ENABLE_TFT": "{TRE:2,TRE_SNAPSHOT_STEPS:5}"})
    run_msrun_command(test_case, log_dir)
    verify_workers_passed(log_dir)
    print(SUCCESS_MESSAGE)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark="allcards", essential_mark='essential')
def test_precision_spike_step_recovery_05() -> None:
    '''
    Feature: precision spike step recovery
    Description: Test precision spike step recovery with HCCE and TRE enabled (TRE_SNAPSHOT_STEPS=5)
                 Verify loss consistency between step10 and 15, and all workers pass
    Expectation: Loss at step10 equals step15, and 8 workers all pass
    '''
    test_case = "test_precision_spike_step_recovery_05"
    log_dir = f"{BASE_PATH}/{test_case}_log"
    setup_log_directory(log_dir)
    setup_environment({"MS_ENABLE_TFT": "{HCCE:1,TRE:2,TRE_SNAPSHOT_STEPS:5}"})
    run_msrun_command(test_case, log_dir)
    loss_lines = extract_loss_values(log_dir)
    lineA1, lineA2 = loss_lines[9], loss_lines[14]
    assert lineA1 == lineA2, f"Loss inconsistency: step10={lineA1}, step15={lineA2}"
    verify_workers_passed(log_dir)
    print(SUCCESS_MESSAGE)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark="allcards", essential_mark='essential')
def test_precision_spike_step_recovery_06() -> None:
    '''
    Feature: precision spike step recovery
    Description: Test precision spike step recovery with TRE enabled (TRE_SNAPSHOT_STEPS=5)
                 Verify loss consistency between step19 and 20, and all workers pass
    Expectation: Loss at step19 equals step20, and 8 workers all pass
    '''
    test_case = "test_precision_spike_step_recovery_06"
    log_dir = f"{BASE_PATH}/{test_case}_log"
    setup_log_directory(log_dir)
    setup_environment({"MS_ENABLE_TFT": "{TRE:2,TRE_SNAPSHOT_STEPS:5}"})
    run_msrun_command(test_case, log_dir)
    loss_lines = extract_loss_values(log_dir)
    lineA1, lineA2 = loss_lines[18], loss_lines[19]
    assert lineA1 == lineA2, f"Loss inconsistency: step19={lineA1}, step20={lineA2}"
    verify_workers_passed(log_dir)
    print(SUCCESS_MESSAGE)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark="allcards", essential_mark='essential')
def test_precision_spike_step_recovery_07() -> None:
    '''
    Feature: precision spike step recovery
    Description: Test precision spike step recovery with TRE enabled (TRE_SNAPSHOT_STEPS=1)
                 Verify loss consistency between step254 and 255, and all workers pass
    Expectation: Loss at step254 equals step255, and 8 workers all pass
    '''
    test_case = "test_precision_spike_step_recovery_07"
    log_dir = f"{BASE_PATH}/{test_case}_log"
    setup_log_directory(log_dir)
    setup_environment({"MS_ENABLE_TFT": "{TRE:2,TRE_SNAPSHOT_STEPS:1}"})
    run_msrun_command(test_case, log_dir)
    loss_lines = extract_loss_values(log_dir)
    lineA1, lineA2 = loss_lines[253], loss_lines[254]
    assert lineA1 == lineA2, f"Loss inconsistency: step254={lineA1}, step255={lineA2}"
    verify_workers_passed(log_dir)
    print(SUCCESS_MESSAGE)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
