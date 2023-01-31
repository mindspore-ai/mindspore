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
import subprocess
from subprocess import Popen

import pytest


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
def test_alloc_memory_fail():
    """
    Feature: The format of exception when memory alloc failed
    Description: Adding two large tensors together causes a memory exception
    Expectation: Throw exception that contains at least 2 message blocks
    """
    cur_path = os.path.split(os.path.realpath(__file__))[0]
    proc = Popen(["pytest", "-sv", os.path.join(cur_path, "alloc_memory_fail.py")],
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.wait()
    stdout_log = proc.stdout.read().decode()
    assert proc.returncode != 0
    pattern = re.compile(r"\nE\s+\-{20,}\nE\s+\- .+\nE\s+\-{20,}\nE\s+.+")
    matches = pattern.findall(stdout_log)
    assert len(matches) >= 2
