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

"""test nccl allreduce performance with 8p"""

import os
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore import context
from mindspore.communication.management import init

np.random.seed(1)
os.environ['GLOG_v'] = str(2)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
init()

class AllReduceNet(nn.Cell):
    def __init__(self):
        super(AllReduceNet, self).__init__()
        self.mul = P.Mul()
        self.all_reduce = P.AllReduce()
        self.add = P.Add()

    def construct(self, x):
        x = self.mul(x, 2)
        y1 = Tensor(np.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]])).astype(np.float32)
        z = self.add(x, y1)
        z = self.all_reduce(z)
        y2 = Tensor(np.array([[-16, -16, -16, -16], [-16, -16, -16, -16], [-16, -16, -16, -16]])).astype(np.float32)
        out = self.add(z, y2)
        out = self.all_reduce(out)
        out = self.mul(out, 2)
        return out

def test_pynative_nccl_allreduce_8p():
    net = AllReduceNet()
    input_x = np.ones([3, 4]).astype(np.float32)
    expect_output = [[256, 256, 256, 256], [256, 256, 256, 256], [256, 256, 256, 256]]
    output = net(Tensor(input_x, mstype.float32))
    assert np.allclose(output.asnumpy(), expect_output)
