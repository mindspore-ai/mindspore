# Copyright 2023 Huawei Technologies Co., Ltd
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
import subprocess
import pytest


def run_same_network_twice_in_one_process(file_name, log_file_name):
    # Clear compile cache folder and log files
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    assert not os.path.exists(log_file_name)

    # First run without compile cache
    cmd_first = f"GLOG_v=1 python " + file_name + " > " + log_file_name + " 2>&1"
    subprocess.check_output(cmd_first, shell=True)
    assert os.path.exists(log_file_name)
    with open(log_file_name, "r") as f_first:
        data_first = f_first.read()

    assert "Generate a new compile key for new args, key: 0" in data_first
    assert "Generate a new compile key for new args, key: 1" not in data_first

    # Clean files
    os.remove(log_file_name)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mutable_compile_repeat():
    """
    Feature: Repeating compile .
    Description: If arg set as mutable(dynamic_len=True) , the different length list args should not cause repeating
    compile.
    Expectation: Network only compile once.
    """
    run_same_network_twice_in_one_process("repeat_compile_mutable_script.py", "repeat_compile_mutable.log")
