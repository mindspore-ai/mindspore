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
""" test if function"""
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_if_function():
    class Net(nn.Cell):
        def __init__(self, func):
            super(Net, self).__init__()
            self.func = func

        def construct(self, x, y):
            if self.func:
                return self.func(x, y)
            return x - y
    def add(x, y):
        return x + y
    net = Net(add)
    x = Tensor(np.ones([1, 2, 3], np.int32))
    y = Tensor(np.ones([1, 2, 3], np.int32))
    net(x, y)
