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
"""GE backend dump test."""

import subprocess
from tests.mark_utils import arg_mark
from tests.security_utils import security_off_wrap


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@security_off_wrap
def test_ge_backend_dump():
    """
    Feature: ge backend dump.
    Description: Test ge backend dump functionality.
    Expectation: Check for expected dump logs in the output.
    """
    script_path = "./run.sh"

    result = subprocess.run(
        ["bash", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
        check=False
    )

    output = result.stdout
    print(output)
    assert "Call Mock function: MS_DbgOnStepBegin" in output
    assert "Call Mock function: MS_DbgOnStepEnd" in output
    assert ("For 'Dump', in the scenario where 'backend' is 'GE', "
            "the 'MINDSPORE_DUMP_CONFIG' env has been deprecated since MindSpore 2.6.") in output
    assert "For 'Dump', the 'ENABLE_MS_GE_DUMP' env has been deprecated since MindSpore 2.6." in output
