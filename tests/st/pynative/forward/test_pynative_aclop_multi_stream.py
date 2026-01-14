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

"""Test PyNative aclop mutli-stream"""

import os
import subprocess
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential',
          mem_peak=4.00)
def test_pynative_aclop_multi_stream():
    '''
    Feature: run allreduce op in pynative mode using msrun.
    Description: Test case entry allreduce op in pynative mode.
    Expectation: Run success.
    '''
    os.environ["VLOG_v"] = "(12900,12910)"
    LOG_FILE = "multi_stream.log"
    return_code = os.system(
        f"pytest -sv test_multi_stream.py::test_pynative_aclop_multi_stream &> {LOG_FILE}"
    )
    assert return_code == 0

    cmd = f'grep AllocTensorMem {LOG_FILE} | grep \'stream id: 2\' | wc -l'
    result = subprocess.check_output(cmd, shell=True, text=True)
    count = int(result.strip())
    assert count == 2, f"Expected allocate memory count 2, got {count}"
