# Copyright 2020 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_full_ps_lenet_ascend():
    """
    Feature: Parameter Server.
    Description: Test LeNet accuracy in ps mode for Ascend.
    Expectation: success.
    """
    return_code = os.system(
        "bash shell_run_test.sh Ascend /home/workspace/mindspore_dataset/mnist 1 1 127.0.0.1 8082"
    )
    if return_code != 0:
        os.system(f"echo '\n**************** Worker Log ****************'")
        os.system(f"grep -E 'ERROR|Error|error' ./worker*/worker*.log")
        os.system(f"echo '\n**************** Server Log ****************'")
        os.system(f"grep -E 'ERROR|Error|error' ./server*/server*.log")
        os.system(f"echo '\n**************** Scheduler Log ****************'")
        os.system(f"grep -E 'ERROR|Error|error' ./sched/sched.log")
    assert return_code == 0


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_full_ps_lenet_gpu():
    """
    Feature: Parameter Server.
    Description: Test LeNet accuracy in ps mode for GPU.
    Expectation: success.
    """
    return_code = os.system(
        "bash shell_run_test.sh GPU /home/workspace/mindspore_dataset/mnist 1 1 127.0.0.1 8082"
    )
    if return_code != 0:
        os.system(f"echo '\n**************** Worker Log ****************'")
        os.system(f"grep -E 'ERROR|Error|error' ./worker*/worker*.log")
        os.system(f"echo '\n**************** Server Log ****************'")
        os.system(f"grep -E 'ERROR|Error|error' ./server*/server*.log")
        os.system(f"echo '\n**************** Scheduler Log ****************'")
        os.system(f"grep -E 'ERROR|Error|error' ./sched/sched.log")
    assert return_code == 0
