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
import stat
import numpy as np
import pytest
import mindspore as ms
from mindspore import context
from mindspore import ops, nn
from mindspore.train.serialization import save_checkpoint

_cur_dir = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
def test_load_checkpoint_for_remove_parameter_prefix():
    """
    Feature: Check the load_checkpoint for remove parameter prefix.
    Description: Check the load_checkpoint in prefix parameter name.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    parameter_list = []
    param1 = {}
    param2 = {}
    param3 = {}
    param4 = {}
    ms.set_seed(1)
    param1['name'] = "0.weight"
    param1['data'] = ms.Tensor([[-0.26484808, -1.3031056],
                                [0.0712087, 0.64198005]])
    param2['name'] = "0.bias"
    param2['data'] = ms.Tensor([-0.73621184, -1.8354928])
    param3['name'] = "1.weight"
    param3['data'] = ms.Tensor([[0.99923426, 0.46487913],
                                [-1.876466, 0.3856378]])
    param4['name'] = "1.bias"
    param4['data'] = ms.Tensor([1.9551374, 0.0045558])
    parameter_list.append(param1)
    parameter_list.append(param2)
    parameter_list.append(param3)
    parameter_list.append(param4)

    if os.path.exists('./parameters.ckpt'):
        os.chmod('./parameters.ckpt', stat.S_IWRITE)
        os.remove('./parameters.ckpt')

    ckpt_file_name = os.path.join(_cur_dir, './parameters.ckpt')
    save_checkpoint(parameter_list, ckpt_file_name)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.cell_list = nn.CellList()

            m1 = nn.Dense(2, 2)
            m2 = nn.Dense(2, 2)
            self.cell_list.append(m1)
            self.cell_list.append(m2)

        def construct(self, x1, x2):
            a = self.cell_list[0](x1)
            b = self.cell_list[0](x2)
            return a + b

    param_dict = ms.load_checkpoint("./parameters.ckpt")

    x1 = ops.ones((2, 2), ms.float32)
    x2 = ops.ones((2, 2), ms.float32)
    net = Net()
    ms.load_param_into_net(net, param_dict)
    out2 = net(x1, x2)
    expect_out2 = np.array([[-4.608331, -2.2446082],
                            [-4.608331, -2.2446082]])
    assert np.allclose(out2.asnumpy(), expect_out2)
