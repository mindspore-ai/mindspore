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
from tests.mark_utils import arg_mark


def run_watch_dde_network(file_name, log_file_name):
    _cur_dir = os.path.dirname(os.path.realpath(__file__))
    file_name = os.path.join(_cur_dir, file_name)
    assert os.path.exists(file_name)

    log_file_name = os.path.join(_cur_dir, log_file_name)
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    assert not os.path.exists(log_file_name)
    cmd_first = f"GLOG_v=2 python " + file_name + " > " + log_file_name + " 2>&1"
    subprocess.check_output(cmd_first, shell=True)
    assert os.path.exists(log_file_name)
    with open(log_file_name, "r") as f_first:
        data_first = f_first.read()
    assert "Purify elements failed" not in data_first

    # Clean files
    os.remove(log_file_name)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_watch_dde_error_log():
    """
    Feature: DDE.
    Description: Some error raised in DDE process unexpected, so add this case to watch it.
    Expectation: No error raised in DDE process .
    """
    run_watch_dde_network("./watch_dde_error_log.py", "watch_dde_error_log.log")
