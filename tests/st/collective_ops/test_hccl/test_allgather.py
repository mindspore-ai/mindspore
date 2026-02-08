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

"""test hccl AllGather and all_gather with 8p"""

import os
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore.communication.management import init
from mindspore.communication import GlobalComm
from mindspore.communication.comm_func import all_gather_into_tensor
from mindspore import context

np.random.seed(1)
context.set_context(jit_level='O0')
os.environ['HCCL_WHITELIST_DISABLE'] = str(1)
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
init()


class AllGatherNet(nn.Cell):
    def __init__(self):
        super(AllGatherNet, self).__init__()
        self.all_gather = P.AllGather()

    def construct(self, x):
        return self.all_gather(x)


class AllGatherFuncNet(nn.Cell):
    def __init__(self, group=GlobalComm.WORLD_COMM_GROUP):
        super(AllGatherFuncNet, self).__init__()
        self.group = group

    def construct(self, x):
        out, _ = all_gather_into_tensor(x)
        return out


def test_hccl_all_gather_into_tensor_8p():
    """
    Feature: test 'AllGather' communication operation.
    Description: test 'AllGather' communication operation.
    Expectation: expect correct result.
    """
    x = np.ones([3, 4]).astype(np.float32)
    net = AllGatherNet()
    expect_output = np.ones([24, 4]).astype(np.float32)
    output = net(Tensor(x, mstype.float32))
    assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_all_gather_into_tensor_func_in_cell_8p():
    """
    Feature: test 'all_gather_into_tensor' communication function in cell.
    Description: test 'all_gather_into_tensor' communication function in cell.
    Expectation: expect correct result.
    """
    x = np.ones([3, 4]).astype(np.float32)
    net = AllGatherFuncNet()
    expect_output = np.ones([24, 4]).astype(np.float32)
    output = net(Tensor(x, mstype.float32))
    assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_all_gather_into_tensor_func_8p():
    """
    Feature: test 'all_gather_into_tensor' communication function.
    Description: test 'all_gather_into_tensor' communication function.
    Expectation: expect correct result.
    """
    x = np.ones([3, 4]).astype(np.float32)
    expect_output = np.ones([24, 4]).astype(np.float32)
    output, _ = all_gather_into_tensor(Tensor(x, mstype.float32))
    assert np.allclose(output.asnumpy(), expect_output)
