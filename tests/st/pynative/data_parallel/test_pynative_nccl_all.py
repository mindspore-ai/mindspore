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
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_gpu'],
          level_mark='level1',
          card_mark='allcards',
          essential_mark='unessential')
def test_pynative_nccl_allreduce():
    return_code = os.system("mpirun -n 8 pytest -s test_pynative_nccl_allreduce.py")
    assert return_code == 0
