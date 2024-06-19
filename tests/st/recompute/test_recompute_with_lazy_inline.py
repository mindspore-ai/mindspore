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
import re
import subprocess
import pytest

from tests.st.utils import test_utils

match_dyn_mem = re.compile(r'Total Static Memory size: (.*?)M', re.S)


def get_max(mem_uses):
    max_mem = 0
    for i in mem_uses:
        max_mem = max(max_mem, int(i))
    return max_mem


def run_testcase(testcase_name, expect_memory_usage):
    # Clear log file
    log_filename = testcase_name + ".log"
    if os.path.exists(log_filename):
        os.remove(log_filename)
    assert not os.path.exists(log_filename)

    cmd = f"export GLOG_v=1; pytest -s test_recompute.py::" + testcase_name + " > " \
          + log_filename + " 2>&1"
    subprocess.check_output(cmd, shell=True)
    assert os.path.exists(log_filename)
    with open(log_filename, "r") as f:
        data = f.read()
    mem_uses = re.findall(match_dyn_mem, data)
    assert len(mem_uses) == 2
    max_mem = get_max(mem_uses)
    assert max_mem == expect_memory_usage
    # Clear log file
    os.remove(log_filename)


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_recompute_cell_recompute():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by the cell recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_block_recompute", 33)


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@test_utils.run_test_with_On
def test_recompute_op_recompute1():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by the primitive recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_op_recompute1", 45)


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_recompute_op_recompute2():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by the primitive recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_op_recompute2", 19)


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_recompute_op_recompute3():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by the primitive recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_op_recompute3", 112)


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_recompute_cell_and_op_recompute1():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by both the primitive and cell recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_cell_and_op_recompute1", 45)


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_recompute_cell_and_op_recompute2():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by both the primitive and cell recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_cell_and_op_recompute2", 51)


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@test_utils.run_test_with_On
def test_recompute_cell_and_op_recompute_with_tuple_outputs1():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by both the primitive and cell recompute api and return a tuple.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_cell_and_op_recompute_with_tuple_outputs1", 53)


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_recompute_cell_and_op_recompute_with_tuple_outputs2():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by both the primitive and cell recompute api and return a tuple.
    Expectation: Run successfully and the memory usage is reduced.
    """
    run_testcase("test_recompute_cell_and_op_recompute_with_tuple_outputs2", 53)
