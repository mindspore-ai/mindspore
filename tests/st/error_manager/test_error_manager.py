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
from tests.mark_utils import arg_mark
import os
import re
import subprocess
from subprocess import Popen


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hccl_init_fail():
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(f"sh {sh_path}/run_hccl_init_fail.sh")
    assert ret == 0
    grep_ret = os.system(f"grep 'Ascend Error Message' {sh_path}/test_hccl_init_fail.log -c")
    assert grep_ret == 0


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_aicerror_message():
    """
    Feature: Improve usability
    Description: Test log when an aicerror occurs
    Expectation: print related log like "[ERROR] Task DebugString, Tbetask..."
    """
    os.environ["GLOG_v"] = "3"
    cur_path = os.path.split(os.path.realpath(__file__))[0]
    proc = Popen(["pytest", os.path.join(cur_path, "aicore_error.py")], stdout=subprocess.PIPE)
    try:
        output, _ = proc.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        proc.kill()
        output, _ = proc.communicate()
    log = output.decode()
    pattern = re.compile(r"\[ERROR\].*?Task DebugString.*?[Aicpu|Tbe|Hccl]task")
    res = pattern.findall(log)
    assert len(res) >= 1
    del os.environ["GLOG_v"]
