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

select = P.Select()
maximum = P.Maximum()
minimum = P.Minimum()
greater = P.Greater()
real_div = P.RealDiv()
mul = P.Mul()
sub = P.Sub()
lamb_update_with_lr = Primitive('LambUpdateWithLR')
make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_lamb_update_with_lr_rule_fusion(tag):
    """ test_lamb_update_with_lr_rule_fusion """
    fns = FnDict()

    @fns
    def before(input0, input1, input2, input3, input4, input5, constant_greater01_max0, constant_select01,
               constant_minimum0):
        real_div_res = real_div(input1, input2)
        greater0_res = greater(input0, constant_greater01_max0)
        greater1_res = greater(input1, constant_greater01_max0)
        res = select(greater0_res, real_div_res, constant_select01)
        res = select(greater1_res, res, constant_select01)
        res = minimum(res, constant_minimum0)
        res = maximum(res, constant_greater01_max0)
        res = mul(res, input3)
        res = mul(res, input4)
        res = sub(input5, res)
        return res

    @fns
    def after(input0, input1, input2, input3, input4, input5, constant_greater01_max0, constant_select01,
              constant_minimum0):
        res = lamb_update_with_lr(input0, input1, input2, input3, input4, input5, constant_greater01_max0,
                                  constant_select01, constant_minimum0)
        return make_tuple(res)

    return fns[tag]
