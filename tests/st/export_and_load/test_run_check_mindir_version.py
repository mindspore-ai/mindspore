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

"""test run check mindir version script."""

import os
import subprocess
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1.00
          )
def test_run_check_mindir_version():
    """
    Feature: check and warn mindir version
    Description: Test check and warn mindir version
    Expectation: No exception and result is correct
    """
    expected_major_warning_count = 2
    expected_minor_warning_count = 1

    script_file = os.path.realpath(os.path.dirname(__file__)) + "/check_mindir_version.py"
    log_file = os.path.realpath(os.path.dirname(__file__)) + "/check_mindir_version.log"

    command = "export GLOG_v=2 && python " + script_file + " > " + log_file + " 2>&1"
    os.system(command)

    try:
        major_warning_count = int(subprocess.check_output(
            f"grep 'Cross-major version compatibility is not guaranteed, "
            f"which may cause model load failure or functional errors.' {log_file} | wc -l",
            shell=True
        ).decode().strip())

        minor_warning_count = int(subprocess.check_output(
            f"grep 'The model may rely on new features not supported by the current "
            f"framework, which may cause functional errors.' {log_file} | wc -l",
            shell=True
        ).decode().strip())

        assert major_warning_count == expected_major_warning_count, \
            f"Expected {expected_major_warning_count}, got {major_warning_count}"
        assert minor_warning_count == expected_minor_warning_count, \
            f"Expected {expected_minor_warning_count}, got {minor_warning_count}"

    except subprocess.CalledProcessError as e:
        pytest.fail(f"Failed to analyse logs: {str(e)}")
    finally:
        if os.path.exists(log_file):
            os.remove(log_file)
