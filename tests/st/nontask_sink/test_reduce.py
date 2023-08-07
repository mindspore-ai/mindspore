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

import os
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _inner_ops as inner_p
from mindspore.communication.management import init, get_rank

# 'Reduce' operator only supports KernelByKernel mode by now. So we set 'GRAPH_OP_RUN' to 1.
np.random.seed(1)
os.environ['GRAPH_OP_RUN'] = str(1)
init()
this_rank = get_rank()

class ReduceNet(nn.Cell):
    def __init__(self):
        super(ReduceNet, self).__init__()
        self.reduce1 = inner_p.Reduce(2)
        self.reduce2 = inner_p.Reduce(6)

    def construct(self, x):
        output1 = self.reduce1(x)
        output2 = self.reduce2(x)
        return output1, output2

def test_hccl_reduce_8p():
    """
    Feature: test 'Reduce' communication operator.
    Description: test 'Reduce' communication operator.
    Expectation: expect correct result.
    """
    net = ReduceNet()
    input_x = np.ones([2, 3, 4, 5]).astype(np.float32)
    expect_output = np.ones([2, 3, 4, 5]).astype(np.float32) * 8
    output1, output2 = net(Tensor(input_x))
    if this_rank == 2:
        assert np.allclose(output1.asnumpy(), expect_output)

    if this_rank == 6:
        assert np.allclose(output2.asnumpy(), expect_output)
    print("outputs are", output1, output2)
