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

Add = P.Add()
Mul = P.Mul()
Sqrt = P.Sqrt()
Square = P.Square()
make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)
LambNextRight = Primitive('LambNextRight')


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_lamb_next_right_rule(tag):
    fns = FnDict()

    @fns
    def before(input0, input1, mul2_x, mul3_x, true_div1_recip, add2_y):
        square0 = Square(input0)
        mul2 = Mul(mul2_x, input1)
        mul3 = Mul(mul3_x, square0)
        add1 = Add(mul2, mul3)
        real_div1 = Mul(add1, true_div1_recip)
        sqrt0 = Sqrt(real_div1)
        add2 = Add(sqrt0, add2_y)
        outputs = make_tuple(add1, add2)
        output = tuple_getitem(outputs, 0)
        return output

    @fns
    def before_unmatched(input0, input1, mul2_x, mul3_x, true_div1_recip, add2_y):
        square0 = Square(input0)
        mul2 = Mul(mul2_x, input1)
        mul3 = Add(mul3_x, square0)
        add1 = Add(mul2, mul3)
        real_div1 = Mul(add1, true_div1_recip)
        sqrt0 = Sqrt(real_div1)
        add2 = Add(sqrt0, add2_y)
        outputs = make_tuple(add1, add2)
        output = tuple_getitem(outputs, 0)
        return output

    @fns
    def after(input0, input1, mul2_x, mul3_x, true_div1_recip, add2_y):
        lamb_next_right = LambNextRight(input0, input1, mul2_x, mul3_x, true_div1_recip, add2_y)
        outputs = make_tuple(tuple_getitem(lamb_next_right, 0), tuple_getitem(lamb_next_right, 1))
        output = tuple_getitem(outputs, 0)
        return make_tuple(output)

    return fns[tag]
