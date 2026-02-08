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

"""test hccl get_process_group_ranks with 8p"""

import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.communication.management import init, get_process_group_ranks
from mindspore import context
from mindspore.communication import GlobalComm

np.random.seed(1)
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
init()


class GroupRanksNet(nn.Cell):
    def __init__(self, group=GlobalComm.WORLD_COMM_GROUP):
        super(GroupRanksNet, self).__init__()
        self.group = group

    def construct(self, x):
        return get_process_group_ranks()


def test_hccl_get_process_group_ranks_func_net_8p():
    """
    Feature: test 'get_process_group_rank' communication function in cell.
    Description: test 'get_process_group_rank' communication function in cell.
    Expectation: expect correct result.
    """
    net = GroupRanksNet()
    test_tensor = np.ones([3, 4]).astype(np.float32)
    output = net(Tensor(test_tensor, mstype.float32))
    expend_output = [0, 1, 2, 3, 4, 5, 6, 7]
    assert np.allclose(output, expend_output)
    print("process_group_ranks output is", output)


def test_hccl_get_process_group_ranks_func_8p():
    """
    Feature: test 'get_process_group_rank' communication function.
    Description: test 'get_process_group_rank' communication function.
    Expectation: expect correct result.
    """
    output = get_process_group_ranks()
    expend_output = [0, 1, 2, 3, 4, 5, 6, 7]
    assert np.allclose(output, expend_output)
    print("process_group_ranks output is", output)
