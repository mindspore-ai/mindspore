# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
""" test super"""
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


class FatherNet(nn.Cell):
    def __init__(self, x):
        super(FatherNet, self).__init__(x)
        self.x = x

    def construct(self, x, y):
        return self.x * x

    def test_father(self, x):
        return self.x + x


class MatherNet(nn.Cell):
    def __init__(self, y):
        super(MatherNet, self).__init__()
        self.y = y

    def construct(self, x, y):
        return self.y * y

    def test_mather(self, y):
        return self.y + y


class SingleSubNet(FatherNet):
    def __init__(self, x, z):
        super(SingleSubNet, self).__init__(x)
        self.z = z

    def construct(self, x, y):
        ret_father_construct = super().construct(x, y)
        ret_father_test = super(SingleSubNet, self).test_father(x)
        ret_father_x = super(SingleSubNet, self).x
        ret_sub_z = self.z

        return ret_father_construct, ret_father_test, ret_father_x, ret_sub_z


class MulSubNet(FatherNet, MatherNet):
    def __init__(self, x, y, z):
        super(MulSubNet, self).__init__(x)
        super(FatherNet, self).__init__(y)
        self.z = z

    def construct(self, x, y):
        ret_father_construct = super().construct(x, y)
        ret_father_test = super(MulSubNet, self).test_father(x)
        ret_father_x = super(MulSubNet, self).x
        ret_mather_construct = super(FatherNet, self).construct(x, y)
        ret_mather_test = super(FatherNet, self).test_mather(y)
        ret_mather_y = super(FatherNet, self).y
        ret_sub_z = self.z

        return ret_father_construct, ret_father_test, ret_father_x, \
               ret_mather_construct, ret_mather_test, ret_mather_y, ret_sub_z


def test_single_super():
    single_net = SingleSubNet(2, 3)
    x = Tensor(np.ones([1, 2, 3], np.int32))
    y = Tensor(np.ones([1, 2, 3], np.int32))
    single_net(x, y)


def test_mul_super():
    mul_net = MulSubNet(2, 3, 4)
    x = Tensor(np.ones([1, 2, 3], np.int32))
    y = Tensor(np.ones([1, 2, 3], np.int32))
    mul_net(x, y)


def test_single_super_in():
    class FatherNetIn(nn.Cell):
        def __init__(self, x):
            super(FatherNetIn, self).__init__(x)
            self.x = x

        def construct(self, x, y):
            return self.x * x

        def test_father(self, x):
            return self.x + x

    class SingleSubNetIN(FatherNetIn):
        def __init__(self, x, z):
            super(SingleSubNetIN, self).__init__(x)
            self.z = z

        def construct(self, x, y):
            ret_father_construct = super().construct(x, y)
            ret_father_test = super(SingleSubNetIN, self).test_father(x)
            ret_father_x = super(SingleSubNetIN, self).x

            return ret_father_construct, ret_father_test, ret_father_x

    single_net_in = SingleSubNetIN(2, 3)
    x = Tensor(np.ones([1, 2, 3], np.int32))
    y = Tensor(np.ones([1, 2, 3], np.int32))
    single_net_in(x, y)
