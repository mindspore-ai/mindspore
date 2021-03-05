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
from mindspore.ops import _constants as Constants

lars_v2 = Primitive('LarsV2')
square_sum_all = Primitive('SquareSumAll')
lars_v2_update = Primitive('LarsV2Update')
make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_lars_v2_fission(tag):
    fns = FnDict()

    @fns
    def before(input0, input1, input2, input3):
        res = lars_v2(input0, input1, input2, input3)
        return res

    @fns
    def after(input0, input1, input2, input3):
        res = square_sum_all(input0, input1)
        item0 = tuple_getitem(res, 0)
        item1 = tuple_getitem(res, 1)
        res = lars_v2_update(input0, input1, item0, item1, input2, input3)
        return make_tuple(res)

    return fns[tag]
