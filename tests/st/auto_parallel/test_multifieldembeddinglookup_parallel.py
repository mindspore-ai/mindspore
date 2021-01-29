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
import os
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_sit_multifieldembeddinglookup_parallel():
    cmd = "mpirun -n 8 pytest -s multifieldembeddinglookup_parallel.py > multifieldembeddinglookup.log 2>&1"
    ret = os.system(cmd)
    os.system(f"grep -E 'ERROR|error' multifieldembeddinglookup.log -C 3")
    assert ret == 0
