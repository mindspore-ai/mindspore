# Copyright 2025 Huawei Technologies Co., Ltd
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
import numpy as np

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, jit
from tests.mark_utils import arg_mark


class BNnet(nn.Cell):
    def __init__(self):
        super(BNnet, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=3, use_batch_statistics=True)

    @jit
    def construct(self, x):
        return self.bn(x)


class Addnet(nn.Cell):
    def __init__(self):
        super(Addnet, self).__init__()
        self.add = ops.AssignAdd()

    @jit
    def construct(self, param, x):
        return self.add(param, x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=1024.02)
def test_format_transmit_batchnorm():
    """
    Feature: special format
    Description: test special format(5hd) transmit of BN parameter(gamma etc.)
    Expectation: no exception
    """
    x = Tensor(np.ones([1, 3, 2, 2]).astype(np.float32))
    x2 = Tensor(np.ones([3]).astype(np.float32))
    bn_net = BNnet()
    _ = bn_net(x)
    add_net = Addnet()
    _ = add_net(bn_net.bn.gamma, x2)
