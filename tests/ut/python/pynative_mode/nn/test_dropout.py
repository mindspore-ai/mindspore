# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test_dropout """
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore.common.api import _executor

context.set_context(device_target="Ascend")


def test_check_dropout_1():
    x = Tensor(np.ones([20, 16, 50]), mstype.float32)
    m = nn.Dropout(0.8)
    m(x)


def test_check_dropout_2():
    x = Tensor(np.ones([20, 16, 50]), mstype.float32)
    m = nn.Dropout(0.3, seed0=1)
    m(x)


def test_check_dropout_3():
    x = Tensor(np.ones([20, 16, 50]), mstype.float32)
    m = nn.Dropout(0.3, seed0=1, seed1=1)
    m(x)


class Net_Dropout(nn.Cell):
    def __init__(self):
        super(Net_Dropout, self).__init__()
        self.dropout = nn.Dropout(0.5)

    def construct(self, x):
        return self.dropout(x)


def test_compile_dropout():
    net = Net_Dropout()
    input_data = Tensor(np.ones([20, 16, 50], dtype=np.float32))
    _executor.compile(net, input_data)
