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

add = P.Add()
mul = P.Mul()
real_div = P.RealDiv()
rsqrt = P.Rsqrt()
sqrt = P.Sqrt()
make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)
LambNextMVWithDecayV1 = Primitive('LambNextMVWithDecayV1')


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_lamb_next_mv_with_decay_v1_rule(tag):
    fns = FnDict()

    @fns
    def before(input0, input1, input2, input3, input4, input5, input6, mul0_x, mul1_sub, mul2_x, mul3_sub1, mul4_x,
               add2_y):
        mul3 = mul(mul3_sub1, input0)
        mul2 = mul(mul2_x, input1)
        add1 = add(mul2, mul3)
        real_div1 = real_div(add1, input2)
        add2 = add(real_div1, add2_y)
        sqrt1 = sqrt(real_div1)
        mul0 = mul(mul0_x, input4)
        mul1 = mul(mul1_sub, input3)
        sqrt0 = rsqrt(add2)
        add4 = add(sqrt1, add2_y)
        add0 = add(mul0, mul1)
        real_div0 = real_div(add0, input5)
        real_div2 = mul(real_div0, sqrt0)
        real_div4 = real_div(real_div0, add4)
        mul4 = mul(mul4_x, input6)
        add3 = add(real_div2, mul4)
        add5 = add(real_div4, mul4)
        res = make_tuple(add3, add0, add1, add5)
        return res

    @fns
    def after(input0, input1, input2, input3, input4, input5, input6, mul0_x, mul1_sub, mul2_x, mul3_sub1, mul4_x,
              add2_y):
        fusion = LambNextMVWithDecayV1(input0, input1, input2, input3, input4, input5, input6, mul0_x, mul1_sub, mul2_x,
                                       mul3_sub1, mul4_x, add2_y)
        item0 = tuple_getitem(fusion, 0)
        item1 = tuple_getitem(fusion, 1)
        item2 = tuple_getitem(fusion, 2)
        item3 = tuple_getitem(fusion, 3)
        res = make_tuple(item0, item1, item2, item3)
        return make_tuple(res)

    @fns
    def no_match1(input0, input1, input2, input3, input4, input5, input6, mul0_x, mul1_sub, mul2_x, mul3_sub1, mul4_x,
                  add2_y):
        mul3 = mul(mul3_sub1, input0)
        mul2 = mul(mul2_x, input1)
        add1 = add(mul2, mul3)
        real_div1 = real_div(add1, input2)
        add2 = add(real_div1, add2_y)
        sqrt1 = sqrt(real_div1)
        mul0 = mul(mul0_x, input4)
        mul1 = mul(mul1_sub, input3)
        sqrt0 = rsqrt(add2)
        add4 = add(sqrt1, add2_y)
        add0 = add(mul0, mul1)
        real_div0 = real_div(add0, input5)
        real_div2 = mul(real_div0, sqrt0)
        real_div4 = real_div(real_div0, add4)
        mul4 = mul(mul4_x, input6)
        add3 = add(real_div2, mul4)
        # diff mul from original add
        add5 = mul(real_div4, mul4)
        res = make_tuple(add3, add0, add1, add5)
        return res

    @fns
    def no_match2(input0, input1, input2, input3, input4, input5, input6, mul0_x, mul1_sub, mul2_x, mul3_sub1, mul4_x,
                  add2_y):
        mul3 = mul(mul3_sub1, input0)
        mul2 = mul(mul2_x, input1)
        add1 = add(mul2, mul3)
        real_div1 = real_div(add1, input2)
        add2 = add(real_div1, add2_y)
        sqrt1 = sqrt(real_div1)
        mul0 = mul(mul0_x, input4)
        mul1 = mul(mul1_sub, input3)
        sqrt0 = rsqrt(add2)
        # diff mul from original add
        add4 = mul(sqrt1, add2_y)
        add0 = add(mul0, mul1)
        real_div0 = real_div(add0, input5)
        real_div2 = mul(real_div0, sqrt0)
        real_div4 = real_div(real_div0, add4)
        mul4 = mul(mul4_x, input6)
        add3 = add(real_div2, mul4)
        add5 = add(real_div4, mul4)
        res = make_tuple(add3, add0, add1, add5)
        return res

    @fns
    def no_match3(input0, input1, input2, input3, input4, input5, input6, mul0_x, mul1_sub, mul2_x, mul3_sub1, mul4_x,
                  add2_y):
        mul3 = mul(mul3_sub1, input0)
        mul2 = mul(mul2_x, input1)
        add1 = add(mul2, mul3)
        real_div1 = real_div(add1, input2)
        add2 = add(real_div1, add2_y)
        # diff sqrt(add2) link from sqrt(real_div1)
        sqrt1 = sqrt(add2)
        mul0 = mul(mul0_x, input4)
        mul1 = mul(mul1_sub, input3)
        sqrt0 = rsqrt(add2)
        add4 = add(sqrt1, add2_y)
        add0 = add(mul0, mul1)
        # diff add from original real_div
        real_div0 = add(add0, input5)
        real_div2 = mul(real_div0, sqrt0)
        real_div4 = real_div(real_div0, add4)
        mul4 = mul(mul4_x, input6)
        add3 = add(real_div2, mul4)
        add5 = add(real_div4, mul4)
        res = make_tuple(add3, add0, add1, add5)
        return res

    return fns[tag]
