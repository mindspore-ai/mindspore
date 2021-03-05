# Copyright 2019 Huawei Technologies Co., Ltd
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

add = P.Add()
sub = P.Sub()
make_tuple = Primitive('MakeTuple')
four2five = Primitive('Four2Five')
five2four = Primitive('Five2Four')
transdata = Primitive("TransData")
cast = Primitive('Cast')
depend = P.Depend()


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_eliminate_5to4_4to5(tag):
    fns = FnDict()

    @fns
    def before(x, y):
        sum_add = add(x, y)
        res = sub(sum_add, y)
        output = make_tuple(res)
        return output

    @fns
    def after1(x, y):
        new_x_sum = transdata(x)
        new_y_sum = transdata(y)
        new_y_sum2 = transdata(y)
        sum_add = add(new_x_sum, new_y_sum)
        sum_5to4 = transdata(sum_add)
        sum_4to5 = transdata(sum_5to4)
        res = sub(sum_4to5, new_y_sum2)
        output = transdata(res)
        new_output = make_tuple(output)
        ret = make_tuple(new_output)
        return ret

    @fns
    def after2(x, y):
        new_x_sum = transdata(x)
        new_y_sum = transdata(y)
        new_y_diff = transdata(y)
        sum_add = add(new_x_sum, new_y_sum)
        res = sub(sum_add, new_y_diff)
        output = transdata(res)
        new_output = make_tuple(output)
        ret = make_tuple(new_output)
        return ret

    return fns[tag]


def test_eliminate_cast(tag):
    fns = FnDict()

    @fns
    def before(x, y):
        sum_add = add(x, y)
        res = sub(sum_add, y)
        output = make_tuple(res)
        return output

    @fns
    def after1(x, y):
        new_x_sum = cast(x)
        new_y_sum = cast(y)
        new_y_sum2 = cast(y)
        sum_add = add(new_x_sum, new_y_sum)
        sum_cast1 = cast(sum_add)
        sum_cast2 = cast(sum_cast1)
        res = sub(sum_cast2, new_y_sum2)
        output = cast(res)
        new_output = make_tuple(output)
        ret = make_tuple(new_output)
        return ret

    @fns
    def after2(x, y):
        new_x_sum = cast(x)
        new_y_sum = cast(y)
        new_y_diff = cast(y)
        sum_add = add(new_x_sum, new_y_sum)
        res = sub(sum_add, new_y_diff)
        output = cast(res)
        new_output = make_tuple(output)
        ret = make_tuple(new_output)
        return ret

    return fns[tag]


def test_eliminate_5to4_depend_4to5(tag):
    fns = FnDict()

    @fns
    def before(x, y):
        sum_add = add(x, y)
        sum_depend = depend(sum_add, x)
        res = sub(sum_depend, y)
        output = make_tuple(res)
        return output

    @fns
    def after1(x, y):
        new_x_sum = transdata(x)
        new_y_sum = transdata(y)
        sum_add = add(new_x_sum, new_y_sum)
        sum_trans = transdata(sum_add)
        depend_between_trans = depend(sum_trans, x)
        depend_trans = transdata(depend_between_trans)
        new_y_diff = transdata(y)
        res = sub(depend_trans, new_y_diff)
        output = transdata(res)
        new_output = make_tuple(output)
        ret = make_tuple(new_output)
        return ret

    @fns
    def after2(x, y):
        new_x_sum = transdata(x)
        new_y_sum = transdata(y)
        sum_add = add(new_x_sum, new_y_sum)
        depend_op = depend(sum_add, x)
        new_y_diff = transdata(y)
        res = sub(depend_op, new_y_diff)
        output = transdata(res)
        new_output = make_tuple(output)
        ret = make_tuple(new_output)
        return ret

    return fns[tag]


def test_eliminate_cast_depend_cast(tag):
    fns = FnDict()

    @fns
    def before(x, y):
        sum_add = add(x, y)
        sum_depend = depend(sum_add, x)
        sum_depend2 = depend(sum_depend, x)
        sum_depend3 = depend(sum_depend2, x)
        res = sub(sum_depend3, y)
        output = make_tuple(res)
        return output

    @fns
    def after1(x, y):
        new_x_sum = cast(x)
        new_y_sum = cast(y)
        sum_add = add(new_x_sum, new_y_sum)
        sum_cast = cast(sum_add)
        depend_between_cast = depend(sum_cast, x)
        depend_between_cast2 = depend(depend_between_cast, x)
        depend_between_cast3 = depend(depend_between_cast2, x)
        depend_cast = cast(depend_between_cast3)
        new_y_diff = cast(y)
        res = sub(depend_cast, new_y_diff)
        output = cast(res)
        new_output = make_tuple(output)
        ret = make_tuple(new_output)
        return ret

    @fns
    def after2(x, y):
        new_x_sum = cast(x)
        new_y_sum = cast(y)
        sum_add = add(new_x_sum, new_y_sum)
        depend_op = depend(sum_add, x)
        depend_op2 = depend(depend_op, x)
        depend_op3 = depend(depend_op2, x)
        new_y_diff = cast(y)
        res = sub(depend_op3, new_y_diff)
        output = cast(res)
        new_output = make_tuple(output)
        ret = make_tuple(new_output)
        return ret

    return fns[tag]
