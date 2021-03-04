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

make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)
square = P.Square()
reduce_sum = P.ReduceSum()
square_sumv1 = Primitive('SquareSumV1')
square_sumv2 = Primitive('SquareSumV2')
axis = -1


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_square_sum_fusion(tag):
    """ test_square_sum_fusion """
    fns = FnDict()

    @fns
    def before1(x):
        square_output = square(x)
        res = reduce_sum(square_output, axis)
        return res

    @fns
    def after1(x):
        res = square_sumv1(x)
        return make_tuple(res)

    @fns
    def before2(x):
        square_output = square(x)
        sum_output = reduce_sum(square_output, axis)
        res = make_tuple(sum_output, square_output)
        return res

    @fns
    def after2(x):
        output = square_sumv2(x)
        item0 = tuple_getitem(output, 0)
        item1 = tuple_getitem(output, 1)
        res = make_tuple(item0, item1)
        return make_tuple(res)

    return fns[tag]
