# Copyright 2021 Huawei Technologies Co., Ltd
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
import re
import shutil
import subprocess
import pytest
import numpy as np

match_output = re.compile(r'[{](.*?)[}]', re.S)
match_num = re.compile(r'\d+\.?\d*', re.S)


def run_twice_with_same_network(file_name, cache_path, log_file_name_first, log_file_name_second):
    # Clear compile cache folder and log files
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    if os.path.exists(log_file_name_first):
        os.remove(log_file_name_first)
    if os.path.exists(log_file_name_second):
        os.remove(log_file_name_second)
    assert not os.path.exists(cache_path)
    assert not os.path.exists(log_file_name_first)
    assert not os.path.exists(log_file_name_second)

    # First run without compile cache
    cmd_first = f"GLOG_v=2 python " + file_name + " '" + cache_path + "' > " + log_file_name_first + " 2>&1"
    subprocess.check_output(cmd_first, shell=True)
    assert os.path.exists(log_file_name_first)
    assert os.path.exists(cache_path)
    with open(log_file_name_first, "r") as f_first:
        data_first = f_first.read()
    assert "Check the consistency of dependency files hash failed. Execute all the compilation actions." in data_first

    # Take out the result of the first run
    match_output_first = re.findall(match_output, data_first)
    assert len(match_output_first) == 2
    nums_first = re.findall(match_num, match_output_first[0])
    array_first = np.array([float(x) for x in nums_first])
    shape_first = re.findall(match_num, match_output_first[1])
    array_shape_first = np.array([int(x) for x in shape_first])

    # Second run with compile cache
    cmd_second = cmd_first = f"GLOG_v=2 python " + file_name + " '" + cache_path + "' > " + log_file_name_second +\
                             " 2>&1"
    subprocess.check_output(cmd_second, shell=True)
    assert os.path.exists(log_file_name_second)
    with open(log_file_name_second, "r") as f_second:
        data_second = f_second.read()
    assert "Use the compilation cache and execute the backend actions only. Be aware of correctness risks." in \
           data_second

    # Take out the result of the second run
    match_output_second = re.findall(match_output, data_second)
    assert len(match_output_second) == 2
    nums_second = re.findall(match_num, match_output_second[0])
    array_second = np.array([float(x) for x in nums_second])
    shape_second = re.findall(match_num, match_output_second[1])
    array_shape_second = np.array([int(x) for x in shape_second])

    assert np.allclose(array_first, array_second, 0.0001, 0.0001)
    assert (array_shape_first == array_shape_second).all()

    # Clean files
    os.remove(log_file_name_first)
    os.remove(log_file_name_second)
    shutil.rmtree(cache_path)


def run_twice_with_different_networks(file_name_first, file_name_second, cache_path, log_file_name_first,
                                      log_file_name_second):
    # Clear compile cache folder
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    assert not os.path.exists(cache_path)

    # First run without compile cache
    cmd_first = f"GLOG_v=2 python " + file_name_first + " '" + cache_path + "' > " + log_file_name_first + " 2>&1"
    subprocess.check_output(cmd_first, shell=True)
    assert os.path.exists(log_file_name_first)
    assert os.path.exists(cache_path)
    with open(log_file_name_first, "r") as f_first:
        data_first = f_first.read()
    assert "Check the consistency of dependency files hash failed. Execute all the compilation actions." in data_first

    # Second run with compile cache
    cmd_second = f"GLOG_v=2 python " + file_name_second + " '" + cache_path + "' > " + log_file_name_second + " 2>&1"
    subprocess.check_output(cmd_second, shell=True)
    assert os.path.exists(log_file_name_second)
    with open(log_file_name_second, "r") as f_second:
        data_second = f_second.read()
    assert "Check the consistency of dependency files hash failed. Execute all the compilation actions." in data_second

    # Clean log files
    os.remove(log_file_name_first)
    os.remove(log_file_name_second)
    shutil.rmtree(cache_path)

@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_compile_cache_load_weights():
    """
    Feature: Compile cache.
    Description: Test whether the compile cache can load the value of parameters successfully.
    Expectation: success.
    """
    run_twice_with_same_network("run_network_with_weights.py", "./weight", "weight_first.txt", "weight_second.txt")


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_compile_cache_lenet():
    """
    Feature: Compile cache.
    Description: Test whether the regular compile cache function can run successfully.
    Expectation: success.
    """
    run_twice_with_same_network("run_lenet.py", "./lenet", "lenet_first.txt", "lenet_second.txt")


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_compile_cache_net_with_control_flow():
    """
    Feature: Compile cache.
    Description: Test whether the compile cache can load ref type parameter correctly.
    Expectation: success.
    """
    run_twice_with_same_network("run_network_with_control_flow.py", "./control_flow", "control_net_first.txt",
                                "control_net_second.txt")


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_compile_cache_auto_detect():
    """
    Feature: Compile cache.
    Description: Test whether the compile cache auto-detection function can run successfully.
    Expectation: success.
    """
    run_twice_with_different_networks("run_lenet.py", "run_network_with_weights.py", "./lenet_auto_detect",
                                      "auto_detect_first.txt", "auto_detect_second.txt")
