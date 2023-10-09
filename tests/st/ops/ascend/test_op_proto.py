# Copyright 2023 Huawei Technologies Co., Ltd
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

import pytest
import subprocess


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_op_proto_warnings():
    """
    Feature: check no warnings produced in op_proto.cc.
    Description: check whether there are warnings produced in op_proto.cc.
    Expectation: no warnings produced in op_proto.cc
    """
    s = subprocess.Popen("python", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE, shell=True)
    s.stdin.write(b"import mindspore as ms\n")
    s.stdin.write(b"ms.set_context(device_target='Ascend')\n")
    s.stdin.write(b"ms.run_check()\n")
    s.stdin.close()

    out = s.stdout.read().decode("UTF-8")
    s.stdout.close()

    op_proto_warnings = []
    lines = out.split('\n')
    for line in lines:
        if line.find('[WARNING]') != 0:
            continue
        if line.find('/op_proto.cc:') > 0:
            op_proto_warnings.append(line)

    if op_proto_warnings:
        print("Unexpeced warnings in op_proto.cc:\n")
        for text in op_proto_warnings:
            print("    " + text)
    assert not op_proto_warnings
