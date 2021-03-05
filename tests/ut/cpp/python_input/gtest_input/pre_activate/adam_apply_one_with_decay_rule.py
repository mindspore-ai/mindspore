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
from mindspore.ops import functional as F
from mindspore.ops import _constants as Constants

mul = P.Mul()
add = P.Add()
square = P.Square()
sqrt = P.Sqrt()
real_div = P.RealDiv()
sub = P.Sub()
Assign = P.Assign()
make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)
adam_apply_one_with_decay = Primitive('AdamApplyOneWithDecay')
adam_apply_one_with_decay_assign = Primitive('AdamApplyOneWithDecayAssign')


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_adam_apply_one_with_decay_rule(tag):
    fns = FnDict()

    @fns
    def before_cond1(input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y):
        mul0 = mul(mul0_x, input2)
        mul1 = mul(mul1_x, input0)
        square0 = square(input0)
        add0 = add(mul0, mul1)
        mul2 = mul(mul2_x, input1)
        mul3 = mul(mul3_x, square0)
        add1 = add(mul2, mul3)
        sqrt0 = sqrt(add1)
        add2 = add(add2_y, sqrt0)
        mul4 = mul(mul4_x, input3)
        real_div0 = real_div(add0, add2)
        add3 = add(mul4, real_div0)
        mul5 = mul(input4, add3)
        sub0 = sub(input3, mul5)
        return make_tuple(add1, add0, sub0)

    @fns
    def before_cond2(input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y):
        mul0 = mul(input2, mul0_x)
        mul1 = mul(input0, mul1_x)
        square0 = square(input0)
        add0 = add(mul0, mul1)
        mul2 = mul(input1, mul2_x)
        mul3 = mul(mul3_x, square0)
        add1 = add(mul2, mul3)
        sqrt0 = sqrt(add1)
        add2 = add(sqrt0, add2_y)
        mul4 = mul(input3, mul4_x)
        real_div0 = real_div(add0, add2)
        add3 = add(mul4, real_div0)
        mul5 = mul(add3, input4)
        sub0 = sub(input3, mul5)
        return make_tuple(add1, add0, sub0)

    @fns
    def before_cond3(input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y):
        mul0 = mul(mul0_x, input2)
        mul1 = mul(mul1_x, input0)
        square0 = square(input0)
        add0 = add(mul0, mul1)
        mul2 = mul(mul2_x, input1)
        mul3 = mul(square0, mul3_x)
        add1 = add(mul2, mul3)
        sqrt0 = sqrt(add1)
        add2 = add(sqrt0, add2_y)
        mul4 = mul(mul4_x, input3)
        real_div0 = real_div(add0, add2)
        add3 = add(mul4, real_div0)
        mul5 = mul(add3, input4)
        sub0 = sub(input3, mul5)
        return make_tuple(add1, add0, sub0)

    @fns
    def before_cond4(input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y):
        mul0 = mul(mul0_x, input2)
        mul1 = mul(mul1_x, input0)
        square0 = square(input0)
        add0 = add(mul0, mul1)
        mul2 = mul(mul2_x, input1)
        mul3 = mul(mul3_x, square0)
        add1 = add(mul2, mul3)
        sqrt0 = sqrt(add1)
        add2 = add(add2_y, sqrt0)
        mul4 = mul(mul4_x, input3)
        real_div0 = real_div(add0, add2)
        add3 = add(mul4, real_div0)
        mul5 = mul(add3, input4)
        sub0 = sub(input3, mul5)
        return make_tuple(add1, add0, sub0)

    @fns
    def before_cond5(input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y):
        mul0 = mul(mul0_x, input2)
        mul1 = mul(mul1_x, input0)
        square0 = square(input0)
        add0 = add(mul0, mul1)
        mul2 = mul(mul2_x, input1)
        mul3 = mul(mul3_x, square0)
        add1 = add(mul2, mul3)
        sqrt0 = sqrt(add1)
        add2 = add(sqrt0, add2_y)
        mul4 = mul(mul4_x, input3)
        real_div0 = real_div(add0, add2)
        add3 = add(mul4, real_div0)
        mul5 = mul(add3, input4)
        sub0 = sub(input3, mul5)
        return make_tuple(add1, add0, sub0)

    @fns
    def after(input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y):
        res = adam_apply_one_with_decay(input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, mul4_x,
                                        add2_y)
        item0 = tuple_getitem(res, 0)
        item1 = tuple_getitem(res, 1)
        item2 = tuple_getitem(res, 2)
        return make_tuple(make_tuple(item0, item1, item2))

    return fns[tag]


def test_adam_apply_one_with_decay_assign_rule(tag):
    fns = FnDict()

    @fns
    def before_cond1(input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y):
        mul0 = mul(mul0_x, input2)
        mul1 = mul(mul1_x, input0)
        square0 = square(input0)
        add0 = add(mul0, mul1)
        mul2 = mul(mul2_x, input1)
        mul3 = mul(mul3_x, square0)
        add1 = add(mul2, mul3)
        sqrt0 = sqrt(add1)
        add2 = add(add2_y, sqrt0)
        mul4 = mul(mul4_x, input3)
        real_div0 = real_div(add0, add2)
        add3 = add(mul4, real_div0)
        mul5 = mul(input4, add3)
        sub0 = sub(input3, mul5)
        assign0 = Assign(input3, sub0)
        depend0 = F.depend(sub0, assign0)
        assign1 = Assign(input2, add0)
        depend1 = F.depend(depend0, assign1)
        assign2 = Assign(input1, add1)
        depend2 = F.depend(depend1, assign2)
        return make_tuple(add1, add0, depend2)

    @fns
    def before_cond2(input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y):
        mul0 = mul(input2, mul0_x)
        mul1 = mul(input0, mul1_x)
        square0 = square(input0)
        add0 = add(mul0, mul1)
        mul2 = mul(input1, mul2_x)
        mul3 = mul(mul3_x, square0)
        add1 = add(mul2, mul3)
        sqrt0 = sqrt(add1)
        add2 = add(sqrt0, add2_y)
        mul4 = mul(input3, mul4_x)
        real_div0 = real_div(add0, add2)
        add3 = add(mul4, real_div0)
        mul5 = mul(add3, input4)
        sub0 = sub(input3, mul5)
        assign0 = Assign(input3, sub0)
        depend0 = F.depend(sub0, assign0)
        assign1 = Assign(input2, add0)
        depend1 = F.depend(depend0, assign1)
        assign2 = Assign(input1, add1)
        depend2 = F.depend(depend1, assign2)
        return make_tuple(add1, add0, depend2)

    @fns
    def before_cond3(input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y):
        mul0 = mul(mul0_x, input2)
        mul1 = mul(mul1_x, input0)
        square0 = square(input0)
        add0 = add(mul0, mul1)
        mul2 = mul(mul2_x, input1)
        mul3 = mul(square0, mul3_x)
        add1 = add(mul2, mul3)
        sqrt0 = sqrt(add1)
        add2 = add(sqrt0, add2_y)
        mul4 = mul(mul4_x, input3)
        real_div0 = real_div(add0, add2)
        add3 = add(mul4, real_div0)
        mul5 = mul(add3, input4)
        sub0 = sub(input3, mul5)
        assign0 = Assign(input3, sub0)
        depend0 = F.depend(sub0, assign0)
        assign1 = Assign(input2, add0)
        depend1 = F.depend(depend0, assign1)
        assign2 = Assign(input1, add1)
        depend2 = F.depend(depend1, assign2)
        return make_tuple(add1, add0, depend2)

    @fns
    def before_cond4(input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y):
        mul0 = mul(mul0_x, input2)
        mul1 = mul(mul1_x, input0)
        square0 = square(input0)
        add0 = add(mul0, mul1)
        mul2 = mul(mul2_x, input1)
        mul3 = mul(mul3_x, square0)
        add1 = add(mul2, mul3)
        sqrt0 = sqrt(add1)
        add2 = add(add2_y, sqrt0)
        mul4 = mul(mul4_x, input3)
        real_div0 = real_div(add0, add2)
        add3 = add(mul4, real_div0)
        mul5 = mul(add3, input4)
        sub0 = sub(input3, mul5)
        assign0 = Assign(input3, sub0)
        depend0 = F.depend(sub0, assign0)
        assign1 = Assign(input2, add0)
        depend1 = F.depend(depend0, assign1)
        assign2 = Assign(input1, add1)
        depend2 = F.depend(depend1, assign2)
        return make_tuple(add1, add0, depend2)

    @fns
    def before_cond5(input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y):
        mul0 = mul(mul0_x, input2)
        mul1 = mul(mul1_x, input0)
        square0 = square(input0)
        add0 = add(mul0, mul1)
        mul2 = mul(mul2_x, input1)
        mul3 = mul(mul3_x, square0)
        add1 = add(mul2, mul3)
        sqrt0 = sqrt(add1)
        add2 = add(sqrt0, add2_y)
        mul4 = mul(mul4_x, input3)
        real_div0 = real_div(add0, add2)
        add3 = add(mul4, real_div0)
        mul5 = mul(add3, input4)
        sub0 = sub(input3, mul5)
        assign0 = Assign(input3, sub0)
        depend0 = F.depend(sub0, assign0)
        assign1 = Assign(input2, add0)
        depend1 = F.depend(depend0, assign1)
        assign2 = Assign(input1, add1)
        depend2 = F.depend(depend1, assign2)
        return make_tuple(add1, add0, depend2)

    @fns
    def after(input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y):
        res = adam_apply_one_with_decay_assign(input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x,
                                               mul4_x, add2_y)
        item0 = tuple_getitem(res, 0)
        item1 = tuple_getitem(res, 1)
        item2 = tuple_getitem(res, 2)
        return make_tuple(make_tuple(item0, item1, item2))

    return fns[tag]
