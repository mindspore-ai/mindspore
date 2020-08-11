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
""" test enumerate"""
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE)


def test_equal_two_const_mstype():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.type_base = mstype.float32
            self.type_0 = mstype.float32
            self.type_1 = mstype.float16
            self.type_2 = mstype.int32
            self.type_3 = mstype.tuple_

        def construct(self):
            ret_0 = self.type_0 == self.type_base
            ret_1 = self.type_1 == self.type_base
            ret_2 = self.type_2 == self.type_base
            ret_3 = self.type_3 == self.type_base
            return ret_0, ret_1, ret_2, ret_3

    net = Net()
    assert net() == (True, False, False, False)


def test_equal_two_tensor_mstype():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x, y, z):
            ret_x = x.dtype == mstype.float32
            ret_y = y.dtype == mstype.int32
            ret_z = z.dtype == mstype.bool_
            ret_xy = x.dtype == y.dtype
            ret_xz = x.dtype == z.dtype
            ret_yz = y.dtype == z.dtype
            return ret_x, ret_y, ret_z, ret_xy, ret_xz, ret_yz

    net = Net()
    x = Tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5)), mstype.float32)
    y = Tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5)), mstype.int32)
    z = Tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5)), mstype.bool_)
    assert net(x, y, z) == (True, True, True, False, False, False)
