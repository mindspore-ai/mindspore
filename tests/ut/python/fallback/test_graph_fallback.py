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
""" test numpy ops """
import pytest
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, ms_function, context
from mindspore.ops import functional as F


context.set_context(mode=context.GRAPH_MODE)

# `add_func` is defined in current file.
def add_func(x, y):
    return x + y

@ms_function
def do_increment(i):
    add_1 = F.partial(add_func, 1)
    return add_1(i)

def test_increment():
    a = do_increment(9)
    assert a == 10


@ms_function
def np_fallback_func():
    array_x = [2, 3, 4, 5]
    np_x = np.array(array_x).astype(np.float32)
    me_x = Tensor(np_x)
    me_x = me_x + me_x
    return me_x

@pytest.mark.skip(reason='Graph fallback feature is not supported yet')
def test_np_fallback_func():
    print(np_fallback_func())


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.x = Tensor([2, 3, 4])

    def construct(self):
        x_len = len(self.x)
        for i in range(x_len):
            print(i)
        return x_len

def test_builtins_len():
    net = Net()
    net()
