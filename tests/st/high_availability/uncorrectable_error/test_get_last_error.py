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
Test `aclrtGetLastError` will be called when enable UCE
"""
from tests.mark_utils import arg_mark
import os
import subprocess

def check_output(output, patterns):
    assert output, "Capture output failed!"
    index = 0
    for pattern in patterns:
        index = output.find(pattern, index)
        assert index != -1, "Unexpected output:\n" + output + "\n--- pattern ---\n" + pattern


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_get_last_error():
    """
    Feature: Test UCE related feature
    Description: Test `aclrtGetLastError` will be called when enable UCE.
    Expectation: When call cann api failed and enable UCE, error code got by `aclrtGetLastError` will be printed
    """
    os.environ['MS_ENABLE_TFT'] = '{UCE:1}'
    file_path = os.path.dirname(__file__)
    result = subprocess.run(["python", f"{file_path}/get_last_error.py"],
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0
    patterns = ['[ERROR]', 'Call ascend api <aclnnMulGetWorkspaceSize> in <operator()> at mindspore/',
                'failed, error code [0].']
    check_output(result.stderr, patterns)
