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
from mindspore.common import monad
from mindspore.common.tensor import Tensor
from mindspore.ops import Primitive
from mindspore.ops import operations as P
from mindspore.ops import _constants as Constants
from mindspore.ops import functional as F

Mul = P.Mul()
ApplyMomentum = P.ApplyMomentum()
FusedMulApplyMomentum = Primitive('FusedMulApplyMomentum')
tuple_getitem = Primitive(Constants.kTupleGetItem)
make_tuple = Primitive('MakeTuple')
constant = Tensor(1.0, mstype.float32)


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_momentum_lossscale_fusion(tag):
    fns = FnDict()

    @fns
    def before(input0, input1, input2, input3, input4):
        mul = Mul(constant, input3)
        fused_mul_apply_momentum = ApplyMomentum(input0, input1, input2, mul, input4, monad.U)
        return fused_mul_apply_momentum

    @fns
    def after(input0, input1, input2, input3, input4):
        dep = F.depend(input4, monad.U)
        return make_tuple(tuple_getitem(FusedMulApplyMomentum(input0, input1, input2, input3, dep, constant), 0))

    return fns[tag]
