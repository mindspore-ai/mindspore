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
import subprocess
import pytest
from mindspore import context, nn


def run_watch_init_no_super(file_name, log_file_name):
    _cur_dir = os.path.dirname(os.path.realpath(__file__))
    file_name = os.path.join(_cur_dir, file_name)
    assert os.path.exists(file_name)

    log_file_name = os.path.join(_cur_dir, log_file_name)
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    assert not os.path.exists(log_file_name)
    cmd_first = f"GLOG_v=3 python " + file_name + " > " + log_file_name + " 2>&1"
    subprocess.check_output(cmd_first, shell=True)
    assert os.path.exists(log_file_name)
    with open(log_file_name, "r") as f_first:
        data_first = f_first.read()
    assert "super().__init__()" in data_first
    assert "__del__" in data_first

    # Clean files
    os.remove(log_file_name)


def test_init_no_super1():
    """
    Feature: Support use Cell attribute.
    Description: Some error raised in test_init_no_super_expectation, so add this case to watch it.
    Expectation: No error raised in test_Init_No_Super1 .
    """
    run_watch_init_no_super("./watch_init_no_super.py", "watch_init_no_super.log")


def test_init_no_super2():
    """
    Feature: Support use Cell attribute.
    Description: Test method init no super in instance
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)

    class InitNoSuper(nn.Cell):
        def __init__(self):
            self.a = 1

        def construct(self):
            return self.a

    net = InitNoSuper()
    with pytest.raises(AttributeError) as info:
        net()
    assert "super().__init__()" in str(info.value)
