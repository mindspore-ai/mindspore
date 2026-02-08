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
"""test destroy copy event"""

from tests.mark_utils import arg_mark
import os

import subprocess


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_destroy_copy_event():
    """
    Feature: check no error produced when destroying copy event
    Description: check whether there are errors produced when destroying copy event.
    Expectation: no errors produced when destroying copy event.
    """
    os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '1'
    with subprocess.Popen("python", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE,
                          shell=True) as s:
        s.stdin.write(b"import mindspore as ms\n")
        s.stdin.write(b"ms.set_device('Ascend')\n")
        s.stdin.write(b"ms.run_check()\n")
        s.stdin.close()

        out = s.stdout.read().decode("UTF-8")
        s.stdout.close()

        errors = []
        lines = out.split('\n')
        for line in lines:
            if line.find('[ERROR]') != 0:
                continue
            if line.find('aclrtDestroyEvent') > 0:
                errors.append(line)

        if errors:
            print("Unexpeced errors occurred:\n")
            for text in errors:
                print("    " + text)
        assert not errors
