# Copyright 2019 Huawei Technologies Co., Ltd
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
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops import _constants as Constants
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype

make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)
bn_grad = G.BatchNormGrad(is_training=True)
sync_bn_grad = G.SyncBatchNormGrad()
bn_grad1 = Primitive('BNGrad1')
bn_grad2 = Primitive('BNGrad2')
bn_grad3 = Primitive('BNGrad3')
bn_training_update_grad = Primitive('BNTrainingUpdateGrad')
bn_training_reduce_grad = Primitive('BNTrainingReduceGrad')
allreduce = Primitive('AllReduce')
mul = Primitive('Mul')
mul_value = Tensor(0.5, mstype.float32)


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_bn_grad_split(tag):
    """ test_bn_grad_split """
    fns = FnDict()

    @fns
    def before(i0, i1, i2, i3, i4, i5):
        bn_grad_output = bn_grad(i0, i1, i2, i3, i4, i5)
        item0 = tuple_getitem(bn_grad_output, 0)
        item1 = tuple_getitem(bn_grad_output, 1)
        item2 = tuple_getitem(bn_grad_output, 2)
        output = make_tuple(item0, item1, item2)
        return output

    @fns
    def after1(i0, i1, i2, i3, i4, i5):
        bn_grad1_output = bn_grad1(i0, i1, i3)
        bn_grad1_item0 = tuple_getitem(bn_grad1_output, 0)
        bn_grad1_item1 = tuple_getitem(bn_grad1_output, 1)
        bn_grad1_item2 = tuple_getitem(bn_grad1_output, 2)
        bn_grad2_output = bn_grad2(bn_grad1_item0, bn_grad1_item1, i4, i2)
        bn_grad2_item0 = tuple_getitem(bn_grad2_output, 0)
        bn_grad2_item1 = tuple_getitem(bn_grad2_output, 1)
        bn_grad2_item2 = tuple_getitem(bn_grad2_output, 2)
        bn_grad2_item3 = tuple_getitem(bn_grad2_output, 3)
        bn_grad2_item4 = tuple_getitem(bn_grad2_output, 4)
        bn_grad3_output = bn_grad3(i0, bn_grad2_item2, bn_grad2_item3, bn_grad2_item4, bn_grad1_item2)
        output = make_tuple(bn_grad3_output, bn_grad2_item0, bn_grad2_item1)
        item0 = tuple_getitem(output, 0)
        item1 = tuple_getitem(output, 1)
        item2 = tuple_getitem(output, 2)
        output = make_tuple(item0, item1, item2)
        return make_tuple(output)

    @fns
    def after2(i0, i1, i2, i3, i4, i5):
        bn_update_grad_output = bn_training_update_grad(i0, i1, i3, i4)
        update_item0 = tuple_getitem(bn_update_grad_output, 0)
        update_item1 = tuple_getitem(bn_update_grad_output, 1)
        bn_reduce_grad_output = bn_training_reduce_grad(i0, i1, update_item0, update_item1, i2, i3, i4)
        output = make_tuple(bn_reduce_grad_output, update_item0, update_item1)
        item0 = tuple_getitem(output, 0)
        item1 = tuple_getitem(output, 1)
        item2 = tuple_getitem(output, 2)
        output = make_tuple(item0, item1, item2)
        return make_tuple(output)

    return fns[tag]


def test_sync_bn_grad_split(tag):
    """ test_sync_bn_grad_split """
    fns = FnDict()

    @fns
    def before(i0, i1, i2, i3, i4):
        bn_grad_output = sync_bn_grad(i0, i1, i2, i3, i4)
        item0 = tuple_getitem(bn_grad_output, 0)
        item1 = tuple_getitem(bn_grad_output, 1)
        item2 = tuple_getitem(bn_grad_output, 2)
        output = make_tuple(item0, item1, item2)
        return output

    @fns
    def after(i0, i1, i2, i3, i4):
        bn_update_grad_output = bn_training_update_grad(i0, i1, i3, i4)
        update_output0 = tuple_getitem(bn_update_grad_output, 0)
        update_output1 = tuple_getitem(bn_update_grad_output, 1)
        allreduce_output0 = allreduce(update_output0)
        allreduce_output1 = allreduce(update_output1)
        update_item0 = mul(allreduce_output0, mul_value)
        update_item1 = mul(allreduce_output1, mul_value)
        bn_reduce_grad_output = bn_training_reduce_grad(i0, i1, update_item0, update_item1, i2, i3, i4)
        output = make_tuple(bn_reduce_grad_output, update_item0, update_item1)
        item0 = tuple_getitem(output, 0)
        item1 = tuple_getitem(output, 1)
        item2 = tuple_getitem(output, 2)
        output = make_tuple(item0, item1, item2)
        return make_tuple(output)

    return fns[tag]
