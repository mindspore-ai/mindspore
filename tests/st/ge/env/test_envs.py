# Copyright 2024 Huawei Technologies Co., Ltd
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
import tests.st.ge.ge_test_utils as utils
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ms_format_mode():
    """
    Feature: for MS_FORMAT_MODE
    Description: add net
    Expectation: the result is correct or the check log is not exist
    """
    utils.run_testcase_and_check_log("test_ms_format_mode", "test_ms_format_mode_0",
                                     "GE option: ge.exec.formatMode, value: 0")
    utils.run_testcase_and_check_log("test_ms_format_mode", "test_ms_format_mode_1",
                                     "GE option: ge.exec.formatMode, value: 1")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ms_disable_ref_mode():
    """
    Feature: for MS_DISABLE_REF_MODE
    Description: add net
    Expectation: the result is correct or the check log is not exist
    """
    utils.run_testcase_and_check_log("test_ms_disable_ref_mode", "test_ms_disable_ref_mode_0_graph_mode",
                                     "GE run graph start in ref mode, graph:")
    utils.run_testcase_and_check_log("test_ms_disable_ref_mode", "test_ms_disable_ref_mode_1_graph_mode",
                                     "GE run graph start, graph: ")
    utils.run_testcase("test_ms_disable_ref_mode", "test_ms_disable_ref_mode_0_pynative_mode")
    utils.run_testcase("test_ms_disable_ref_mode", "test_ms_disable_ref_mode_1_pynative_mode")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ms_ascend_check_overflow_mode():
    """
    Feature: for MS_ASCEND_CHECK_OVERFLOW_MODE
    Description: add net
    Expectation: the result is correct or the check log is not exist
    """
    utils.run_testcase_and_check_log("test_ms_ascend_check_overflow_mode", "test_saturation_mode",
                                     "The current overflow detection mode is Saturation")
    utils.run_testcase_and_check_log("test_ms_ascend_check_overflow_mode", "test_infnan_mode",
                                     "The current overflow detection mode is INFNAN")
    utils.run_testcase_and_check_log("test_ms_ascend_check_overflow_mode", "test_unset_condition",
                                     "The current overflow detection mode is INFNAN")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ms_enable_io_reuse():
    """
    Feature: for MS_ENABLE_IO_REUSE
    Description: add net
    Expectation: the result is correct or the check log is not exist
    """
    utils.run_testcase_and_check_log("test_ms_enable_io_reuse", "test_ms_enable_io_reuse_1",
                                     "Enable io reuse: 1")
    utils.run_testcase_and_check_log("test_ms_enable_io_reuse", "test_ms_enable_io_reuse_0",
                                     "Enable io reuse: 0")
