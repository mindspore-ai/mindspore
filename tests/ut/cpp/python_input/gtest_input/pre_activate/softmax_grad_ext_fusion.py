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

Mul = P.Mul()
ReduceSum = P.ReduceSum(keep_dims=True)
Sub = P.Sub()
SoftmaxGradExt = Primitive('SoftmaxGradExt')
MakeTuple = Primitive('MakeTuple')
TupleGetItem = Primitive(Constants.kTupleGetItem)
axes = (2, 3)


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_softmax_grad_ext_fusion(tag):
    fns = FnDict()

    @fns
    def before(input0, input1, input2):
        mul = Mul(input1, input0)
        # input axis will be convert to attr in step ConstructKernelGraph
        reduce_sum = ReduceSum(mul, axes)
        sub = Sub(input0, reduce_sum)
        mul1 = Mul(input2, input1)
        mul_grad = Mul(mul1, sub)
        return mul_grad

    @fns
    def after(input0, input1, input2):
        res = SoftmaxGradExt(input0, input1, input2)
        return MakeTuple(res)

    return fns[tag]


def test_softmax_grad_ext_fusion_v2(tag):
    fns = FnDict()

    @fns
    def before(input0, input1, input2):
        mul = Mul(input1, input0)
        reduce_sum = ReduceSum(mul, axes)
        sub = Sub(input0, reduce_sum)
        mul1 = Mul(input1, sub)
        mul_grad = Mul(input2, mul1)
        return mul_grad

    @fns
    def after(input0, input1, input2):
        res = SoftmaxGradExt(input0, input1, input2)
        return MakeTuple(res)

    return fns[tag]


def test_softmax_grad_ext_fusion_v3(tag):
    fns = FnDict()

    @fns
    def before(input0, input1, input2):
        mul = Mul(input1, input0)
        reduce_sum = ReduceSum(mul, axes)
        sub = Sub(input0, reduce_sum)
        mul1 = Mul(input1, sub)
        mul_grad = Mul(mul1, input2)
        return mul_grad

    @fns
    def after(input0, input1, input2):
        res = SoftmaxGradExt(input0, input1, input2)
        return MakeTuple(res)

    return fns[tag]
