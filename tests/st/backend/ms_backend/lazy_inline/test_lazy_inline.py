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
""" test_lazy_inline.py """
import os
import subprocess
from tests.mark_utils import arg_mark


def run_testcase(testcase_name, expect_sub_str):
    # Clear log file
    log_filename = testcase_name + ".log"
    if os.path.exists(log_filename):
        os.remove(log_filename)
    assert not os.path.exists(log_filename)
    cmd = ("export GLOG_v=1; pytest -s test_lazy_inline_return_tuple.py::") + \
          testcase_name + " > " + log_filename + " 2>&1"
    subprocess.check_output(cmd, shell=True)
    assert os.path.exists(log_filename)
    with open(log_filename, "r", encoding='utf-8') as f:
        data = f.read()
    assert expect_sub_str in data
    # Clear log file
    os.remove(log_filename)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=108.00)
def test_return_tuple():
    """
    Feature: lazy inline return tuple
    Description: test lazy inline return tuple
    Expectation: test pass
    """
    run_testcase("test_return_tuple", "can be lazyinlined")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=138.00)
def test_return_tuple_nest():
    """
    Feature: lazy inline return tuple
    Description: test lazy inline return tuple
    Expectation: test pass
    """
    run_testcase("test_return_tuple_nest", "Set no inline because cell reuse graph has nested make tuple")
