# Copyright 2022 Huawei Technologies Co., Ltd
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
import os
import pytest


@pytest.mark.level2
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_split_ref_without_optim():
    """
    Feature: Graph partition.
    Description: Test splitting one ref node to another process while optimizer is in local process.
    Expectation: success.
    """
    return_code = os.system(
        "bash shell_run_test.sh GPU 2 2 127.0.0.1 8082 true split_ref_without_optim"
    )
    if return_code != 0:
        os.system(f"echo '\n**************** Worker Log ****************'")
        os.system(f"grep -E 'ERROR|Error|error' -C 15 ./worker*/worker*.log")
        os.system(f"echo '\n**************** Scheduler Log ****************'")
        os.system(f"grep -E 'ERROR|Error|error' -C 15 ./sched/sched.log")
    assert return_code == 0


@pytest.mark.level2
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_split_optim():
    """
    Feature: Graph partition.
    Description: Test splitting optimizer nodes to another process while there's side-effect node in local process.
    Expectation: success.
    """
    return_code = os.system(
        "bash shell_run_test.sh GPU 2 2 127.0.0.1 8082 true split_optim"
    )
    if return_code != 0:
        os.system(f"echo '\n**************** Worker Log ****************'")
        os.system(f"grep -E 'ERROR|Error|error' -C 15 ./worker*/worker*.log")
        os.system(f"echo '\n**************** Scheduler Log ****************'")
        os.system(f"grep -E 'ERROR|Error|error' -C 15 ./sched/sched.log")
    assert return_code == 0
