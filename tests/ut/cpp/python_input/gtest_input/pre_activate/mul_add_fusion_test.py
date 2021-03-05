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
fused_mul_add = Primitive('FusedMulAdd')
make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_mul_add_fusion(tag):
    fns = FnDict()

    @fns
    def before1(x, y, z):
        res = mul(x, y)
        res = add(res, z)
        return res

    @fns
    def before2(x, y, z):
        res = mul(x, y)
        res = add(z, res)
        return res

    @fns
    def after(x, y, z):
        res = fused_mul_add(x, y, z)
        return make_tuple(res)

    return fns[tag]
