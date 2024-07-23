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

import os
import platform
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_julia():
    """
    Feature: test custom op of julia cases
    Description: run julia_cases
    Expectation: res == 0
    """
    system = platform.system()
    machine = platform.machine()
    if system != 'Linux' or machine != 'x86_64':
        return
    res = os.system('sh julia_run.sh')
    if res != 0:
        assert False, 'julia test fail'
