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
""" test not in"""
import numpy as np

import mindspore.nn as nn
from mindspore import context, Tensor

context.set_context(mode=context.GRAPH_MODE)


def test_number_not_in_tuple():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.tuple_ = (2, 3, 4)
            self.list_ = [2, 3, 4]
            self.dict_ = {"a": Tensor(np.ones([1, 2, 3], np.int32)),
                          "b": Tensor(np.ones([1, 2, 3], np.int32)),
                          "c": Tensor(np.ones([1, 2, 3], np.int32))}
            self.number_in = 3
            self.number_not_in = 5
            self.str_in = "a"
            self.str_not_in = "e"

        def construct(self):
            ret = 0
            if self.number_in not in self.tuple_:
                ret += 1
            if self.number_not_in not in self.tuple_:
                ret += 2
            if self.number_in not in self.list_:
                ret += 3
            if self.number_not_in not in self.list_:
                ret += 4
            if self.str_in not in self.dict_:
                ret += 5
            if self.str_not_in not in self.dict_:
                ret += 6
            return ret

    net = Net()
    output = net()
    assert output == 12
