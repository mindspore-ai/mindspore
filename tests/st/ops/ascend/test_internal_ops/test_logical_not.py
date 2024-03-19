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

import random
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.common.dtype as mstype


class LogicalNotNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.logical_not = ops.LogicalNot()

    def construct(self, input_x):
        out = self.logical_not(input_x)
        return out


def random_int_list(start, stop, length):
    """
    Feature: random_int_list
    Description: random_int_list
    Expectation: start <= out <= stop
    """
    random_list = []
    for _ in range(length):
        random_list.append(random.randint(start, stop))
    return random_list


def test_logical_not_net():
    """
    Feature: logical_not test case
    Description: logical_not test case
    Expectation: the result is correct
    """
    net = LogicalNotNet()

    size = 10
    np_data_bool = random_int_list(0, 1, size)
    out = ~np.array(np_data_bool).astype(np.bool_)

    ms_in = Tensor(np_data_bool, mstype.bool_)
    ms_out = net(ms_in)
    assert ms_out.dtype == mstype.bool_
    assert np.allclose(ms_out.asnumpy(), out, 0.01, 0.01)
    print("logical_not success.")
