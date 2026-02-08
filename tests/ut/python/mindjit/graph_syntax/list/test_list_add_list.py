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
""" test list add list """

import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.value1 = [Tensor([1, 2, 3]), Tensor([4, 5, 6])]
        self.value2 = [Tensor([7, 8, 9]), Tensor([10, 11, 12])]

    def construct(self):
        return self.value1 + self.value2


def test_list_add_list():
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    expect_ret = (Tensor([1, 2, 3]), Tensor([4, 5, 6]), Tensor([7, 8, 9]), Tensor([10, 11, 12]))
    for i in range(len(net())):
        assert (np.array_equal(net()[i].asnumpy(), expect_ret[i].asnumpy()))
