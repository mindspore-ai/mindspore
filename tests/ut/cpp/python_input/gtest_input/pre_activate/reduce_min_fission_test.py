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
reduce_min = P.ReduceMin(keep_dims=False)
reduce_min1 = Primitive('ReduceMin')
reduce_min2 = Primitive('ReduceMin')


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_reduce_min_fission(tag):
    fns = FnDict()

    @fns
    def before(x):
        res = reduce_min(x, (2, 3))
        return res

    @fns
    def after(x):
        res = reduce_min1(x)
        res = reduce_min2(res)
        return make_tuple(res)

    return fns[tag]
