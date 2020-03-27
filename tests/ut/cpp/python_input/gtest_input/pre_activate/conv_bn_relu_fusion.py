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

from mindspore.ops import operations as P
from mindspore.ops import Primitive

make_tuple = Primitive('make_tuple')
tuple_getitem = Primitive('tuple_getitem')
conv = P.Conv2D(out_channel=64, kernel_size=7, mode=1, pad_mode="valid", pad=0, stride=1, dilation=1, group=1)
bn = P.FusedBatchNorm()
relu = P.ReLU()
conv_bn1 = Primitive('ConvBN1')
bn2_relu = Primitive('BN2Relu')


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_conv_bn_relu_fusion(tag):
    """ test_conv_bn_relu_fusion """
    fns = FnDict()

    @fns
    def before(x, w, scale, b, mean, variance):
        conv_output = conv(x, w)
        bn_output = bn(conv_output, scale, b, mean, variance)
        item0 = tuple_getitem(bn_output, 0)
        item1 = tuple_getitem(bn_output, 3)
        item2 = tuple_getitem(bn_output, 4)
        output = make_tuple(relu(item0), item1, item2)
        res = tuple_getitem(output, 0)
        return res

    @fns
    def after(x, w, scale, b, mean, variance):
        conv_bn1_output = conv_bn1(x, w)
        conv_item0 = tuple_getitem(conv_bn1_output, 0)
        conv_item1 = tuple_getitem(conv_bn1_output, 1)
        conv_item2 = tuple_getitem(conv_bn1_output, 2)
        bn2_relu_output = bn2_relu(conv_item0, conv_item1, conv_item2, scale, b, mean, variance)
        bn2_relu_item0 = tuple_getitem(bn2_relu_output, 0)
        bn2_relu_item1 = tuple_getitem(bn2_relu_output, 1)
        bn2_relu_item2 = tuple_getitem(bn2_relu_output, 2)
        bn2_relu_item3 = tuple_getitem(bn2_relu_output, 3)
        new_make_tuple = make_tuple(bn2_relu_item0, bn2_relu_item1, bn2_relu_item2, conv_item2, bn2_relu_item3)
        item1 = tuple_getitem(new_make_tuple, 3)
        item2 = tuple_getitem(new_make_tuple, 4)
        output = make_tuple(bn2_relu_item0, item1, item2)
        return make_tuple(tuple_getitem(output, 0))

    return fns[tag]
