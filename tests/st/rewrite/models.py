# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Models for test."""
from typing import Optional
from mindspore import Tensor
import mindspore.nn as nn


class BaseNet(nn.Cell):
    def __init__(self, a):
        super().__init__()
        self.relu = nn.ReLU()
        self.a = a

    def construct(self, x: Optional[Tensor]):
        return x

    def add_a(self, x):
        x = x + self.a
        return x


class NoCellNet():
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def no_cell_func(self, x: Optional[Tensor]):
        return x


def external_func(x):
    return x


def external_func2(x):
    return x

EXTERN_LIST = [Tensor(1)]

class NetWithClassVar():
    var1 = Tensor(1.0)
    var2 = external_func
    if True: # pylint: disable=using-constant-test
        var3 = external_func2
    var4 = EXTERN_LIST

    def __init__(self, a):
        self.a = a

    def class_var_func(self, x: Optional[Tensor]):
        # test class variables
        x = x + self.var1
        x = NetWithClassVar.var2(x)
        x = NetWithClassVar.var3(x)
        x = x + NetWithClassVar.var4[0]
        # test instance variable
        x = x + self.a
        return x
