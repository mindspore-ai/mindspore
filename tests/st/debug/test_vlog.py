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
"""
test MindSpore vlog interface
"""

import subprocess
from tests.mark_utils import arg_mark


def check_output(vlog_v, expect_output, is_expect=True):
    """set VLOG_v to vlog_v and check output """
    cmd = f"VLOG_v={vlog_v} python -c 'import mindspore as ms'"
    s = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    out = s.stdout.read().decode("UTF-8")
    s.stdout.close()

    matched = False
    lines = out.split('\n')
    for line in lines:
        if line.find(expect_output) > 0:
            matched = True
            break
    if is_expect:
        assert matched, f'`VLOG_v={vlog_v}` expect `{expect_output}` fail'
    else:
        assert not matched, '`VLOG_v={vlog_v}` unexpected `{expect_output}` fail'


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_op_proto_warnings():
    """
    Feature: test mindspore vlog interface
    Description: check whether mindspore vlog can work properly
    Expectation: vlog can work properly
    """
    # test invalid VLOG_v value
    check_output('xxx', 'Value of environment var VLOG_v is invalid:')
    check_output('2147483648', 'Value of environment var VLOG_v is invalid:')
    check_output('(1,2147483648)', 'Value of environment var VLOG_v is invalid:')
    check_output('(,)', 'Value of environment var VLOG_v is invalid:')
    check_output('0', 'Value of environment var VLOG_v is invalid:')

    # test_valid VLOG_v value and expect some outputs
    check_output('(1,2147483647)', ': log level for printing vlog tags already been used')
    check_output('(,2147483647)', ': log level for printing vlog tags already been used')
    check_output('(1,)', ': log level for printing vlog tags already been used')
    check_output('20000', ': log level for printing vlog tags already been used')

    # test_valid VLOG_v value and unexpected some outputs
    check_output('(5,3)', ': log level for printing vlog tags already been used', False)
    check_output('(,20)', ': log level for printing vlog tags already been used', False)
    check_output('(20001,)', ': log level for printing vlog tags already been used', False)
    check_output('50', ': log level for printing vlog tags already been used', False)
