# Copyright 2020 Huawei Technologies Co., Ltd
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

import pytest
import numpy as np
from mindspore import context
from mindspore.nn import ReLU
from mindspore.nn import Cell
from mindspore.common.tensor import Tensor
from mindspore.common.api import jit

def setup_module():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_pynative_staging_together():
    class NetPynative(Cell):
        def __init__(self):
            super().__init__()
            self.relu = ReLU()
        def construct(self, x):
            return self.relu(x)

    class NetStaging(Cell):
        def __init__(self):
            super().__init__()
            self.relu = ReLU()
        @jit
        def construct(self, x):
            return self.relu(x)

    input1 = np.random.randn(2, 2).astype(np.float32)

    net1 = NetPynative()
    out_me_pynative = net1(Tensor(input1)).asnumpy()

    net2 = NetStaging()
    out_me_staging = net2(Tensor(input1)).asnumpy()

    assert np.allclose(out_me_pynative, out_me_staging, 0.001, 0.001)
