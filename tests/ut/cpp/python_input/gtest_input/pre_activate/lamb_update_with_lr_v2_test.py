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
from mindspore.ops import Primitive
from mindspore.ops import operations as P
from mindspore.ops import _constants as Constants

Sub = P.Sub()
Mul = P.Mul()
RealDiv = P.RealDiv()
Select = P.Select()
Greater = P.Greater()
make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)
LambUpdateWithLrV2 = Primitive('LambUpdateWithLrV2')


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_lamb_update_with_lr_v2(tag):
    fns = FnDict()

    @fns
    def before(input0, input1, input2, input3, input4, greater_y, select_e):
        greater0 = Greater(input0, greater_y)
        greater1 = Greater(input1, greater_y)
        real_div0 = RealDiv(input0, input1)
        select0 = Select(greater1, real_div0, select_e)
        select1 = Select(greater0, select0, select_e)
        mul0 = Mul(select1, input2)
        mul1 = Mul(mul0, input3)
        sub0 = Sub(input4, mul1)
        return sub0

    @fns
    def after(input0, input1, input2, input3, input4, greater_y, select_e):
        res = LambUpdateWithLrV2(input0, input1, input2, input3, input4, greater_y, select_e)
        return make_tuple(res)

    return fns[tag]
