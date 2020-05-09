# Copyright 2019 Huawei Technologies Co., Ltd
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

import re
import numpy as np
from mindspore import context
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor
from tests.ut.python.ops.test_math_ops import VirtualLoss
import mindspore as ms
from mindspore.common.api import _executor
from mindspore.ops import composite as C
from mindspore.parallel._utils import _reset_op_id as reset_op_id

class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x):
        predict = self.network(x)
        return self.loss(predict)

class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x):
        return C.grad_all(self.network)(x)


    # model_parallel test
def test_auto_parallel_assign_sub_with_ref_key():
    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    
    x = Tensor(np.random.rand(4, 4, 32, 64),dtype=ms.float32)

    net = NetWithLoss(nn.PReLU(4))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_auto_parallel()
    reset_op_id()

    _executor.compile(net, x, phase="train")
    strategies = _executor._get_strategy(net)
    for (k, v) in strategies.items():
        if re.search('PReLU-op', k) is not None:
            assert v == [[1, 1, 1, 8], [1]]
        elif re.search('ReLU-op', k) is not None:
            assert v == [[1]]

