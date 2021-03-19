# Copyright 2021 Huawei Technologies Co., Ltd
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

tuple_getitem = Primitive(Constants.kTupleGetItem)
depend = P.Depend()
addn = P.AddN()
add = P.Add()
sub = P.Sub()
mul = P.Mul()
max_pool = P.MaxPoolWithArgmax(pad_mode="same", kernel_size=3, strides=2)
make_tuple = Primitive('MakeTuple')
four2five = Primitive('Four2Five')
five2four = Primitive('Five2Four')
cast = Primitive('Cast')
relu = P.ReLU()


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_merge_cast_input(tag):
    fns = FnDict()

    @fns
    def before(x, y):
        res = addn((x, y))
        return res

    @fns
    def after1(x, y):
        new_x = cast(x)
        new_y = cast(y)
        res = addn(four2five(new_x), four2five(new_y))
        output = cast(five2four(res))
        new_output = make_tuple(output)
        return new_output

    @fns
    def after2(x, y):
        new_x = four2five(x)
        new_y = four2five(y)
        res = addn(new_x, new_y)
        output = cast(five2four(res))
        new_output = make_tuple(output)
        return new_output

    return fns[tag]


def test_merge_cast_output_for_single_output(tag):
    fns = FnDict()

    @fns
    def before(x, y):
        res = addn((x, y))
        return res

    @fns
    def after1(x, y):
        new_x = cast(x)
        new_y = cast(y)
        res = addn(four2five(new_x), four2five(new_y))
        output = cast(five2four(res))
        new_output = make_tuple(output)
        return new_output

    @fns
    def after2(x, y):
        new_x = four2five(cast(x))
        new_y = four2five(cast(y))
        res = addn(new_x, new_y)
        output = five2four(res)
        new_output = make_tuple(output)
        return new_output

    return fns[tag]


def test_merge_cast_output_for_multiple_output(tag):
    fns = FnDict()

    @fns
    def before(x):
        output = max_pool(x)
        res = tuple_getitem(output, 0)
        return res

    @fns
    def after(x):
        output = max_pool(x)
        item0 = tuple_getitem(output, 0)
        item1 = cast(tuple_getitem(output, 1))
        res = make_tuple(item0, item1)
        return make_tuple(tuple_getitem(res, 0))

    return fns[tag]


def test_eliminate_depend_input2(tag):
    fns = FnDict()

    @fns
    def before(x, y, z):
        new_z = four2five(z)
        depend_intput = depend(y, new_z)
        sum_add = add(x, depend_intput)
        return sum_add

    @fns
    def after(x, y, z):
        depend_intput = depend(y, z)
        sum_add = add(x, depend_intput)
        return sum_add

    return fns[tag]


def test_func_graph_cse(tag):
    """ test_func_graph_cse """
    fns = FnDict()

    @fns
    def g1(x, y):
        a = add(x, y)
        b = add(x, y)
        c = mul(a, b)
        return c

    @fns
    def g2(x, y):
        a = add(x, y)
        b = add(mul(a, y), sub(a, x))
        c = add(mul(a, y), sub(add(x, y), x))
        d = add(b, c)
        return d

    return fns[tag]
