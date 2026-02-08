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
Test module for race check.
"""
import os
from tests.mark_utils import arg_mark


def run_command(cmd, log_path, has_error=False):
    if os.path.isfile(log_path):
        os.remove(log_path)
    os.system(cmd)
    error_flag = False
    with open(log_path, 'r') as file:
        for line in file:
            if '[ERROR]' in line:
                error_flag = True
                print("error line: ", line, flush=True)
    assert error_flag == has_error
    if os.path.isfile(log_path):
        os.remove(log_path)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='dryrun_only', essential_mark='essential')
def test_race_check_ok():
    """
    Feature: race check
    Description: race check ok
    Expectation: run success
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    run_command(f"bash {sh_path}/dryrun_race_check.sh race_check_ok.py race_check_ok.log",
                f"{sh_path}/race_check_ok.log", has_error=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='dryrun_only', essential_mark='essential')
def test_race_check_error():
    """
    Feature: race check
    Description: race check error
    Expectation: check race
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    run_command(f"bash {sh_path}/dryrun_race_check.sh race_check_error.py race_check_error.log",
                f"{sh_path}/race_check_error.log", has_error=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='dryrun_only', essential_mark='essential')
def test_race_check_assign():
    """
    Feature: race check
    Description: race check assign
    Expectation: check race
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    run_command(f"bash {sh_path}/dryrun_race_check.sh race_check_assign.py race_check_assign.log",
                f"{sh_path}/race_check_assign.log", has_error=True)
