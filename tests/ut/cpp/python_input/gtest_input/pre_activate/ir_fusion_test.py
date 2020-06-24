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
from mindspore.ops.operations import _grad_ops as G

# pylint: disable=unused-variable

tuple_getitem = Primitive('tuple_getitem')
add = P.TensorAdd()
allreduce = P.AllReduce()
allreduce.add_prim_attr('fusion', 1)
make_tuple = Primitive('make_tuple')
conv = P.Conv2D(out_channel=64, kernel_size=7, mode=1, pad_mode="valid", pad=0, stride=1, dilation=1, group=1)
bn = P.FusedBatchNorm()
relu = P.ReLU()
conv_bn1 = Primitive('ConvBN1')
bn2_add_relu = Primitive('BN2AddRelu')
bn2_relu = Primitive('BN2Relu')
fused_bn1 = Primitive('FusedBN1')
fused_bn2 = Primitive('FusedBN2')
fused_bn3 = Primitive('FusedBN3')
bn_grad = G.FusedBatchNormGrad()
bn_grad1 = Primitive('BNGrad1')
bn_grad2 = Primitive('BNGrad2')
bn_grad3 = Primitive('BNGrad3')


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
        item0 = tuple_getitem(bn_output, 0)
        return item0

    @fns
    def after(x, scale, b, mean, variance):
        fused_bn1_output = fused_bn1(x)
        fused_bn2_input0 = tuple_getitem(fused_bn1_output, 0)
        fused_bn2_input1 = tuple_getitem(fused_bn1_output, 1)
        fused_bn2_output = fused_bn2(fused_bn2_input0, fused_bn2_input1, mean, variance)
        fused_bn3_input1 = tuple_getitem(fused_bn2_output, 0)
        fused_bn3_input2 = tuple_getitem(fused_bn2_output, 1)
        fused_bn3_output = fused_bn3(x, fused_bn3_input1, fused_bn3_input2, scale, b)
        output1 = tuple_getitem(fused_bn2_output, 2)
        output2 = tuple_getitem(fused_bn2_output, 3)
        output3 = tuple_getitem(fused_bn2_output, 0)
        output4 = tuple_getitem(fused_bn2_output, 1)
        output = make_tuple(fused_bn3_output, output1, output2, output3, output4)
        item0 = tuple_getitem(output, 0)
        return make_tuple(item0)

    return fns[tag]


def test_bn_grad_split(tag):
    """ test_bn_grad_split """
    fns = FnDict()

    @fns
    def before(dy, x, scale, save_mean, save_inv_variance):
        bn_grad_output = bn_grad(dy, x, scale, save_mean, save_inv_variance)
        item0 = tuple_getitem(bn_grad_output, 0)
        item1 = tuple_getitem(bn_grad_output, 1)
        item2 = tuple_getitem(bn_grad_output, 2)
        output = make_tuple(item0, item1, item2)
        res = tuple_getitem(output, 0)
        return res

    @fns
    def after(i0, i1, i2, i3, i4):
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
        bn_grad_make_tuple = make_tuple(bn_grad3_output, bn_grad2_item0, bn_grad2_item1)
        item0 = tuple_getitem(bn_grad_make_tuple, 0)
        item1 = tuple_getitem(bn_grad_make_tuple, 1)
        item2 = tuple_getitem(bn_grad_make_tuple, 2)
        output = make_tuple(item0, item1, item2)
        return make_tuple(tuple_getitem(output, 0))

    return fns[tag]


def test_all_reduce_fusion_all(tag):
    """ test_all_reduce_fusion_all """
    fns = FnDict()

    @fns
    def before(x1, x2, x3, x4, x5):
        y1 = allreduce(x1)
        y2 = allreduce(x2)
        y3 = allreduce(x3)
        y4 = allreduce(x4)
        y5 = allreduce(x5)
        return make_tuple(y1, y2, y3, y4, y5)

    @fns
    def after(x1, x2, x3, x4, x5):
        ar = allreduce(x5, x4, x3, x2, x1)
        y5 = tuple_getitem(ar, 0)
        y4 = tuple_getitem(ar, 1)
        y3 = tuple_getitem(ar, 2)
        y2 = tuple_getitem(ar, 3)
        y1 = tuple_getitem(ar, 4)
        res = make_tuple(y1, y2, y3, y4, y5)
        return make_tuple(res)

    @fns
    def after1(x1, x2, x3, x4, x5):
        ar = allreduce(x1, x2, x3, x4, x5)
        y1 = tuple_getitem(ar, 0)
        y2 = tuple_getitem(ar, 1)
        y3 = tuple_getitem(ar, 2)
        y4 = tuple_getitem(ar, 3)
        y5 = tuple_getitem(ar, 4)
        res = make_tuple(y1, y2, y3, y4, y5)
        return make_tuple(res)

    return fns[tag]


def test_all_reduce_fusion_group(tag):
    """ test_all_reduce_fusion_group """
    fns = FnDict()

    @fns
    def before(x1, x2, x3, x4, x5):
        y1 = allreduce(x1)
        y2 = allreduce(x2)
        y3 = allreduce(x3)
        y4 = allreduce(x4)
        y5 = allreduce(x5)
        return make_tuple(y1, y2, y3, y4, y5)

    @fns
    def after1(x1, x2, x3, x4, x5):
        ar1 = allreduce(x5, x4)
        ar2 = allreduce(x3, x2, x1)
        y4 = tuple_getitem(ar1, 1)
        y5 = tuple_getitem(ar1, 0)
        y1 = tuple_getitem(ar2, 2)
        y2 = tuple_getitem(ar2, 1)
        y3 = tuple_getitem(ar2, 0)
        res = make_tuple(y1, y2, y3, y4, y5)
        return make_tuple(res)

    @fns
    def after2(x1, x2, x3, x4, x5):
        ar1 = allreduce(x1, x3, x5)
        ar2 = allreduce(x2, x4)
        y1 = tuple_getitem(ar1, 2)
        y3 = tuple_getitem(ar1, 1)
        y5 = tuple_getitem(ar1, 0)
        y2 = tuple_getitem(ar2, 1)
        y4 = tuple_getitem(ar2, 0)
        output = make_tuple(y1, y2, y3, y4, y5)
        return make_tuple(output)

    return fns[tag]
