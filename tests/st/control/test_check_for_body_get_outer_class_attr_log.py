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
from tests.st.control.cases_register import case_register


@case_register.level1
@case_register.target_gpu
def test_catch_exception_stack_trace_log():
    """
    Feature: Resolve.
    Description: execute the testcase 'for_body_get_outer_class_attr.py::test_catch_exception_of_get_outer_class_attr'
        when MS_DEV_ENABLE_FALLBACK_RUNTIME is set, check the log info.
    Expectation: the error code exist in log info.
    """
    file_name = "for_body_get_outer_class_attr.py"
    log_file_name = "for_body_get_outer_class_attr.log"
    function_name = "::test_catch_exception_of_get_outer_class_attr"
    _cur_dir = os.path.dirname(os.path.realpath(__file__))
    file_name = os.path.join(_cur_dir, file_name)
    assert os.path.exists(file_name)

    log_file_name = os.path.join(_cur_dir, log_file_name)
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    assert not os.path.exists(log_file_name)
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    cmd_first = f"GLOG_v=2 pytest -s " + file_name + function_name + " > " + log_file_name + " 2>&1"
    out = os.popen(cmd_first)
    out.read()
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'
    assert os.path.exists(log_file_name)
    with open(log_file_name, "r") as f_first:
        data_first = f_first.read()
    assert "Do not support to convert" in data_first
    assert "x = self.y.tt1" in data_first

    # Clean files
    os.remove(log_file_name)
