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
# ============================================================================
import os
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('run_mode', [0, 1])
def test_op_debug_option(run_mode):
    """
    Feature: test op debug option
    Description: run_mode is 0 for graph_mode and 1 for pynative
    Expectation: success or throw exception when keyword does not exist in log
    """
    self_path = os.path.split(os.path.realpath(__file__))[0]
    return_code = os.system(f"bash {self_path}/shell_run_test.sh {run_mode}")
    if return_code != 0:
        os.system(f"echo '\n**************** Log ****************'")
        os.system(f"grep -E 'ERROR|Error|error' {self_path}/ms.log")
    os.system(f"rm -rf {self_path}/ms.log")
    assert return_code == 0
