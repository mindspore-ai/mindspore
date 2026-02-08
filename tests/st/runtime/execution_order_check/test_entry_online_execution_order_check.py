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
test_online_execution_order_check
"""
import os
from tests.mark_utils import arg_mark

def test_online_execution_order_check():
    """
    Feature: Test online execution order check.
    Description: Test online execution order check, which includes AllReduce, AllGather.
    Expectation: The program execute and exit normally.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(current_dir, "test_online_execution_order_check.py")

    return_code = os.system(
        f"msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 \
        --master_port=10988 --join=True \
        {script}"
    )
    if return_code != 0:
        os.system("echo '\n**************** Worker Log (Errors) ****************'")
        os.system("grep -E 'ERROR|Error|error|CRITICAL' -C 15 ./worker*.log")
        os.system("echo '\n**************** Worker Log (Last 200 Lines) ****************'")
        os.system("tail -n 200 ./worker*.log")
        os.system("echo '\n**************** Scheduler Log (Errors) ****************'")
        os.system("grep -E 'ERROR|Error|error|CRITICAL' -C 15 ./scheduler.log")
        os.system("echo '\n**************** Scheduler Log (Last 200 Lines) ****************'")
        os.system("tail -n 200 ./scheduler.log")
    assert return_code == 0
