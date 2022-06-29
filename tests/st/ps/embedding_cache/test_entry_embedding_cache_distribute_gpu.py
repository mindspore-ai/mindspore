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


@pytest.mark.level2
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_embedding_cache_distribute_gpu():
    """
    Feature: Test embedding cache feature on gpu with 4 workers, 2 servers.
    Description: 4 workers train network containing embedding layers and enable embedding cache.
    Expectation: All process execute and exit normal.
    """

    self_path = os.path.split(os.path.realpath(__file__))[0]
    return_code = os.system(f"bash {self_path}/run_test_embedding_cache_distribute.sh GPU 4 127.0.0.1 8077 0")
    if return_code != 0:
        os.system(f"echo '\n**************** Worker Log ****************'")
        os.system(f"grep -E 'ERROR|Error|error' {self_path}/worker*/worker*.log")
        os.system(f"echo '\n**************** Server Log ****************'")
        os.system(f"grep -E 'ERROR|Error|error' {self_path}/server*/server*.log")
        os.system(f"echo '\n**************** Scheduler Log ****************'")
        os.system(f"grep -E 'ERROR|Error|error' {self_path}/sched/sched.log")
    assert return_code == 0
