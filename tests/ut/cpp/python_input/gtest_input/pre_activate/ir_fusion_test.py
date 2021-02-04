# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

# pylint: disable=unused-variable

tuple_getitem = Primitive(Constants.kTupleGetItem)
add = P.Add()
allreduce = P.AllReduce()
allreduce.add_prim_attr('fusion', 1)
make_tuple = Primitive("MakeTuple")
conv = P.Conv2D(out_channel=64, kernel_size=7, mode=1, pad_mode="valid", pad=0, stride=1, dilation=1, group=1)
relu = P.ReLU()
conv_bn1 = Primitive('ConvBN1')
bn2_add_relu = Primitive('BN2AddRelu')
bn2_relu = Primitive('BN2Relu')
fused_bn1 = Primitive('FusedBN1')
fused_bn2 = Primitive('FusedBN2')
fused_bn3 = Primitive('FusedBN3')
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
        ar1 = allreduce(x1, x2)
        ar2 = allreduce(x3, x4, x5)
        y1 = tuple_getitem(ar1, 0)
        y2 = tuple_getitem(ar1, 1)
        y3 = tuple_getitem(ar2, 0)
        y4 = tuple_getitem(ar2, 1)
        y5 = tuple_getitem(ar2, 2)
        res = make_tuple(y1, y2, y3, y4, y5)
        return make_tuple(res)

    @fns
    def after2(x1, x2, x3, x4, x5):
        ar1 = allreduce(x1, x3, x5)
        ar2 = allreduce(x2, x4)
        y1 = tuple_getitem(ar1, 0)
        y3 = tuple_getitem(ar1, 1)
        y5 = tuple_getitem(ar1, 2)
        y2 = tuple_getitem(ar2, 0)
        y4 = tuple_getitem(ar2, 1)
        output = make_tuple(y1, y2, y3, y4, y5)
        return make_tuple(output)

    return fns[tag]
