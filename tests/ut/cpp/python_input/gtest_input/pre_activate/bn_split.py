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
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops import _constants as Constants
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype

make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)
bn = P.BatchNorm(is_training=True)
sync_bn = inner.SyncBatchNorm()
fused_bn1 = Primitive('FusedBN1')
fused_bn2 = Primitive('FusedBN2')
fused_bn3 = Primitive('FusedBN3')
bn_training_reduce = Primitive('BNTrainingReduce')
bn_training_update = Primitive('BNTrainingUpdate')
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


def test_bn_split(tag):
    """ test_split_bn_fusion """
    fns = FnDict()

    @fns
    def before(x, scale, b, mean, variance):
        bn_output = bn(x, scale, b, mean, variance)
        output = tuple_getitem(bn_output, 0)
        return output

    @fns
    def after(x, scale, b, mean, variance):
        fused_bn1_output = fused_bn1(x)
        fused_bn2_input0 = tuple_getitem(fused_bn1_output, 0)
        fused_bn2_input1 = tuple_getitem(fused_bn1_output, 1)
        fused_bn2_output = fused_bn2(fused_bn2_input0, fused_bn2_input1, mean, variance)
        fused_bn3_input1 = tuple_getitem(fused_bn1_output, 0)
        fused_bn3_input2 = tuple_getitem(fused_bn2_output, 0)
        fused_bn3_output = fused_bn3(x, fused_bn3_input1, fused_bn3_input2, scale, b)
        output1 = tuple_getitem(fused_bn2_output, 1)
        output2 = tuple_getitem(fused_bn2_output, 2)
        output3 = tuple_getitem(fused_bn1_output, 0)
        output4 = tuple_getitem(fused_bn2_output, 0)
        output = make_tuple(fused_bn3_output, output1, output2, output3, output4)
        output = tuple_getitem(output, 0)
        return make_tuple(output)

    return fns[tag]


def test_bn_split_tbe(tag):
    """ test_split_bn_fusion """
    fns = FnDict()

    @fns
    def before(x, scale, b, mean, variance):
        bn_output = bn(x, scale, b, mean, variance)
        output = tuple_getitem(bn_output, 0)
        return output

    @fns
    def after(x, scale, b, mean, variance):
        bn_training_reduce_output = bn_training_reduce(x)
        bn_training_update_input1 = tuple_getitem(bn_training_reduce_output, 0)
        bn_training_update_input2 = tuple_getitem(bn_training_reduce_output, 1)
        bn_training_update_output = bn_training_update(x, bn_training_update_input1, bn_training_update_input2,
                                                       scale, b, mean, variance)
        output = tuple_getitem(bn_training_update_output, 0)
        return make_tuple(output)

    return fns[tag]


def test_sync_bn_split_tbe(tag):
    """ test_sync_split_bn_fusion """
    fns = FnDict()

    @fns
    def before(x, scale, b, mean, variance):
        bn_output = sync_bn(x, scale, b, mean, variance)
        output = tuple_getitem(bn_output, 0)
        return output

    @fns
    def after(x, scale, b, mean, variance):
        bn_training_reduce_output = bn_training_reduce(x)
        bn_training_reduce_output0 = tuple_getitem(bn_training_reduce_output, 0)
        bn_training_reduce_output1 = tuple_getitem(bn_training_reduce_output, 1)
        allreduce_output0 = allreduce(bn_training_reduce_output0)
        allreduce_output1 = allreduce(bn_training_reduce_output1)
        bn_training_update_input1 = mul(allreduce_output0, mul_value)
        bn_training_update_input2 = mul(allreduce_output1, mul_value)
        bn_training_update_output = bn_training_update(x, bn_training_update_input1, bn_training_update_input2,
                                                       scale, b, mean, variance)
        output = tuple_getitem(bn_training_update_output, 0)
        return make_tuple(output)

    return fns[tag]
