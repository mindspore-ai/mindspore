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
import re
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_compile_remind_msg():
    """
    Feature: Compile start and end remind message print.
    Description: Test start and end remind message.
    Expectation: Run success.
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(f"sh {sh_path}/run.sh")
    assert ret == 0

    file_name = f"{sh_path}/tmp.log"
    assert os.path.exists(file_name)
    with open(os.path.join(file_name), 'r') as f:
        contend = f.read()
        assert len(re.findall(r"EVENT", contend)) == 2
        assert len(re.findall(r"PID", contend)) == 2
        assert len(
            re.findall(r"Start compiling and it will take a while. Please wait...", contend)) == 1
        assert len(re.findall(r"End compiling.", contend)) == 1

    os.remove(file_name)
