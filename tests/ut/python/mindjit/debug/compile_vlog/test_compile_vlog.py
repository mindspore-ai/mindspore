# Copyright 2024-2025 Huawei Technologies Co., Ltd
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


def generate_vlog(file_name, vlog_file_name):
    # Clear compile cache folder and log files
    if os.path.exists(vlog_file_name):
        os.remove(vlog_file_name)
    assert not os.path.exists(vlog_file_name)

    dirname = os.path.dirname(os.path.abspath(__file__))
    cmd = f"VLOG_v=1 python " + dirname + "/" + file_name + " > " + vlog_file_name + " 2>&1"
    subprocess.check_output(cmd, shell=True)
    assert os.path.exists(vlog_file_name)
    with open(vlog_file_name, "r") as v_file:
        data = v_file.read()

    assert "Start compiling" in data
    assert "End compiling" in data

    # Clean files
    os.remove(vlog_file_name)


def test_vlog():
    """
    Feature: Add vlog printing logs.
    Description: If VLOG_v is set to 1, print the ME pipeline logs.
    Expectation: Vlog log successfully printed
    """
    generate_vlog("run_compile_vlog.py", "run_generate_vlog.log")
