# Copyright 2021 Huawei Technologies Co., Ltd
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

"""test CPU data parallel training"""

import os
import sys

from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_train_lenet_by_data_parallel():
    """
    Feature: CPU data parallel.
    Description: Test CPU data parallel training LeNet.
    Expectation: Each node loss is converged and parameters are updated.
    """
    if sys.platform != 'linux':
        return
    return_code = os.system("bash build_lenet_cluster.sh")
    assert return_code == 0
