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

"""test hccl AllReduce and all_reduce with 8p"""

import os
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore.communication.management import init
from mindspore.communication.comm_func import all_reduce
from mindspore import context
from mindspore.ops import ReduceOp

np.random.seed(1)
os.environ['HCCL_WHITELIST_DISABLE'] = str(1)
context.set_context(jit_level='O0')
context.set_context(device_target="Ascend")
init()


class AllReduceNet(nn.Cell):
    def __init__(self):
        super(AllReduceNet, self).__init__()
        self.mul = P.Mul()
        self.all_reduce = P.AllReduce()
        self.add = P.Add()
        self.y1 = Tensor(np.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]])).astype(np.float32)
        self.y2 = Tensor(np.array([[-16, -16, -16, -16], [-16, -16, -16, -16], \
                                   [-16, -16, -16, -16]])).astype(np.float32)

    def construct(self, x):
        x = self.mul(x, 2)
        z = self.add(x, self.y1)
        z = self.all_reduce(z)
        out = self.add(z, self.y2)
        out = self.all_reduce(out)
        out = self.mul(out, 2)
        return out


class AllReduceFuncNet(nn.Cell):
    def __init__(self, op=ReduceOp.SUM):
        super(AllReduceFuncNet, self).__init__()
        self.op = op

    def construct(self, x):
        out, _ = all_reduce(x)
        return out


def test_hccl_allreduce_8p():
    """
    Feature: test 'AllReduce' communication operation.
    Description: test 'AllReduce' communication operation.
    Expectation: expect correct result.
    """
    net = AllReduceNet()
    input_x = np.ones([3, 4]).astype(np.float32)
    expect_output = [[256, 256, 256, 256], [256, 256, 256, 256], [256, 256, 256, 256]]
    output = net(Tensor(input_x, mstype.float32))
    assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_allreduce_func_net_8p():
    """
    Feature: test 'all_reduce' communication function in cell.
    Description: test 'all_reduce' communication function in cell.
    Expectation: expect correct result.
    """
    net = AllReduceFuncNet()
    input_x = np.ones([3, 4]).astype(np.float32)
    expect_output = [[8, 8, 8, 8], [8, 8, 8, 8], [8, 8, 8, 8]]
    output = net(Tensor(input_x, mstype.float32))
    assert np.allclose(output.asnumpy(), expect_output)


def test_hccl_allreduce_func_8p():
    """
    Feature: test 'all_reduce' communication function.
    Description: test 'all_reduce' communication function.
    Expectation: expect correct result.
    """
    x = np.ones([3, 4]).astype(np.float32)
    expect_output = [[8, 8, 8, 8], [8, 8, 8, 8], [8, 8, 8, 8]]
    output, _ = all_reduce(Tensor(x, mstype.float32))
    assert np.allclose(output.asnumpy(), expect_output)
