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
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import Primitive
from mindspore.ops import operations as P
from mindspore.ops import _constants as Constants

addn = P.AddN()
mul = P.Mul()
fused_mul_addn = Primitive('FusedMulAddN')
make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)
scalar = Tensor(1.0, mstype.float32)


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_mul_addn_fusion(tag):
    fns = FnDict()

    @fns
    def before(a, b):
        res = mul(scalar, a)
        res = addn((res, b))
        return res

    @fns
    def after(a, b):
        res = fused_mul_addn(a, b, scalar)
        return make_tuple(res)

    @fns
    def unmatch(a, b, c):
        # none of mul's input is scalar
        res = mul(a, b)
        res = addn((c, res))
        return res

    return fns[tag]
