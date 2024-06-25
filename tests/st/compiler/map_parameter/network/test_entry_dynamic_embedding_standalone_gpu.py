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
import pytest
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_embedding_gpu():
    """
    Feature: Test dynamic embedding feature on gpu.
    Description: A small network contain dynamic embedding(MapParameter).
    Expectation: All process execute and exit normal.
    """

    self_path = os.path.split(os.path.realpath(__file__))[0]
    return_code = os.system(f"bash {self_path}/run_test_dynamic_embedding_standalone.sh GPU")
    if return_code != 0:
        os.system(f"echo '\n**************** Log ****************'")
        os.system(f"grep -E 'ERROR|Error|error' {self_path}/dynamic_embedding.log")
    assert return_code == 0
