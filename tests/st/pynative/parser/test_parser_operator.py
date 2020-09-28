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
""" test_parser_operator """
import pytest
import numpy as np
from mindspore import context
from mindspore.nn import ReLU
from mindspore.nn import Cell
from mindspore.common.tensor import Tensor

def setup_module():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_parser_operator_floor_div():
    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.relu = ReLU()

        def construct(self, x):
            x = self.relu(x)
            x = 3 // x
            return x

    input_np_x = np.array(2).astype(np.float32)
    input_me_x = Tensor(input_np_x)
    net = Net()
    out_me = net(input_me_x)

    assert np.allclose(out_me.asnumpy(), 3 // input_np_x, 0.001, 0.001)
