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

addn = P.AddN()
mul = P.Mul()
reduce_sum = P.ReduceSum()
confusion_mul_grad = Primitive('ConfusionMulGrad')
make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)
axis = 1


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_confusion_mul_grad_fusion(tag):
    fns = FnDict()

    @fns
    def before(input1, input2, input3):
        addn0 = addn((input2, input2))

        output1 = mul(input1, addn0)
        mul1 = mul(input3, addn0)
        # input axis will be convert to attr in step ConstructKernelGraph
        output2 = reduce_sum(mul1, axis)
        res = make_tuple(output1, output2)
        return res

    @fns
    def after(input1, input2, input3):
        addn0 = addn(input2, input2)
        res = confusion_mul_grad(input1, addn0, input3)
        item0 = tuple_getitem(res, 0)
        item1 = tuple_getitem(res, 1)
        res = make_tuple(item0, item1)
        return make_tuple(res)

    return fns[tag]
