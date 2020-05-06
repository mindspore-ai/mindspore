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
from mindspore.ops import operations as P
from mindspore.ops import Primitive
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor

AssignSub = P.AssignSub()
Mul = P.Mul()
Sub = P.Sub()
make_tuple = Primitive('make_tuple')
tuple_getitem = Primitive('tuple_getitem')
depend = Primitive('depend')
BatchNorm = P.BatchNorm()
BNTrainingReduce = Primitive('BNTrainingReduce')
BNTrainingUpdate = Primitive('BNTrainingUpdate')
constant0 = Tensor(0.1, mstype.float32)
constant1 = Tensor(0.1, mstype.float32)


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_fused_batch_norm_fusion(tag):
    fns = FnDict()

    @fns
    def before(input0, input1, input2, input3, input4, var0, var1):
        batch_norm = BatchNorm(input0, input1, input2, input3, input4)
        sub0 = Sub(var0, tuple_getitem(batch_norm, 1))
        sub1 = Sub(var1, tuple_getitem(batch_norm, 2))
        mul0 = Mul(sub0, constant0)
        mul1 = Mul(sub1, constant1)
        assign_sub0 = AssignSub(var0, mul0)
        assign_sub1 = AssignSub(var1, mul1)
        depend0 = depend(tuple_getitem(batch_norm, 0), assign_sub0)
        depend1 = depend(depend0, assign_sub1)
        outputs = make_tuple(depend1, tuple_getitem(batch_norm, 3), tuple_getitem(batch_norm, 4))
        output = tuple_getitem(outputs, 0)
        return output

    @fns
    def after(input0, input1, input2, input3, input4, var0, var1):
        bn_training_reduce = BNTrainingReduce(input0)
        bn_training_update = BNTrainingUpdate(input0, tuple_getitem(bn_training_reduce, 0),
                                              tuple_getitem(bn_training_reduce, 1), input1, input2, var0, var1)
        outputs = make_tuple(tuple_getitem(bn_training_update, 0), tuple_getitem(bn_training_update, 3),
                             tuple_getitem(bn_training_update, 4))
        output = tuple_getitem(outputs, 0)
        return make_tuple(output)

    return fns[tag]
