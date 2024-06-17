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

"""test hccl reduce with 8p"""

import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import comm_ops
from mindspore.communication.management import init, get_rank
from mindspore.communication.comm_func import reduce
from mindspore import context
from mindspore.communication import GlobalComm

# 'Reduce' operator only supports KernelByKernel mode by now.
np.random.seed(1)
context.set_context(jit_level='O0')
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
init()
this_rank = get_rank()


class ReduceNet(nn.Cell):
    def __init__(self):
        super(ReduceNet, self).__init__()
        self.reduce1 = comm_ops.Reduce(2)
        self.reduce2 = comm_ops.Reduce(6)

    def construct(self, x):
        output1 = self.reduce1(x)
        output2 = self.reduce2(x)
        return output1, output2


class ReduceFuncNet(nn.Cell):
    def __init__(self, group=GlobalComm.WORLD_COMM_GROUP):
        super(ReduceFuncNet, self).__init__()
        self.group = group

    def construct(self, x):
        output1 = reduce(x, 2)
        output2 = reduce(x, 6)
        return output1, output2


def test_hccl_reduce_8p():
    """
    Feature: test 'Reduce' communication operator.
    Description: test 'Reduce' communication operator.
    Expectation: expect correct result.
    """
    net = ReduceNet()
    input_x = np.array([0, 1, 2, 3]).astype(np.float32)
    expect_output = np.array([0, 8, 16, 24]).astype(np.float32)
    output1, output2 = net(Tensor(input_x))
    if this_rank == 2:
        assert np.allclose(output1.asnumpy(), expect_output)

    if this_rank == 6:
        assert np.allclose(output2.asnumpy(), expect_output)
    print("outputs are", output1, output2)


def test_hccl_reduce_func_net_8p():
    """
    Feature: test 'Reduce' communication operator.
    Description: test 'Reduce' communication operator.
    Expectation: expect correct result.
    """
    net = ReduceFuncNet()
    input_x = np.array([0, 1, 2, 3]).astype(np.float32)
    expect_output = np.array([0, 8, 16, 24]).astype(np.float32)
    output1, output2 = net(Tensor(input_x))
    if this_rank == 2:
        assert np.allclose(output1.asnumpy(), expect_output)

    if this_rank == 6:
        assert np.allclose(output2.asnumpy(), expect_output)
    print("outputs are", output1, output2)


def test_hccl_reduce_func_8p():
    """
    Feature: test 'reduce' communication function.
    Description: test 'reduce' communication function.
    Expectation: expect correct result.
    """
    input_x = np.array([0, 1, 2, 3]).astype(np.float32)
    expect_output = np.array([0, 8, 16, 24]).astype(np.float32)
    output1 = reduce(Tensor(input_x), 2)
    output2 = reduce(Tensor(input_x), 6)
    if this_rank == 2:
        assert np.allclose(output1.asnumpy(), expect_output)

    if this_rank == 6:
        assert np.allclose(output2.asnumpy(), expect_output)
    print("outputs are", output1, output2)
