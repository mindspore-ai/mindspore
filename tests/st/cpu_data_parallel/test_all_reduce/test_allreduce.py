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

"""test all reduce on CPU"""

import os
import sys

import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_allreduce():
    """
    Feature: CPU data parallel.
    Description: Test AllReduce op on CPU.
    Expectation: Each node obtains all node reduced result.
    """
    if sys.platform != 'linux':
        return
    return_code = os.system("bash build_allreduce_net_cluster.sh run_allreduce.py 8119")
    if return_code != 0:
        os.system(f"echo '\n**************** Worker Log ****************'")
        os.system(f"grep -E 'ERROR|Error|error' -C 15 ./worker*.log")
        os.system(f"echo '\n**************** Scheduler Log ****************'")
        os.system(f"grep -E 'ERROR|Error|error' -C 15 ./scheduler.log")
    assert return_code == 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_allreduce_small_scale_data():
    """
    Feature: CPU data parallel.
    Description: Test allreduce small scale data on CPU.
    Expectation: Each node obtains all node reduced result.
    """
    if sys.platform != 'linux':
        return
    return_code = os.system("bash build_allreduce_net_cluster.sh run_allreduce_small_scale_data.py 8081")
    assert return_code == 0
