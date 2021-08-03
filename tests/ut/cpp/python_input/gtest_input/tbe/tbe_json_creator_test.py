# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G

Relu = P.ReLU()
Cast = P.Cast()
Add = P.Add()
Conv2DBackpropFilter = G.Conv2DBackpropFilter(out_channel=4,
                                              kernel_size=1,
                                              pad_mode="valid",
                                              pad=0,
                                              mode=1,
                                              stride=1,
                                              dilation=1,
                                              group=1)
DynamicRNN = P.DynamicRNN(forget_bias=0.0)
LayerNorm = P.LayerNorm()
Conv2D = P.Conv2D(out_channel=32, kernel_size=3)


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_tbe_json_creator(tag):
    fns = FnDict()

    @fns
    def func_relu_relu_cast(x):
        relu1 = Relu(x)
        relu2 = Relu(relu1)
        res = Cast(relu2, mstype.float16)
        return res

    @fns
    def func_conv2d_backprop_filter(x, out, shape):
        return Conv2DBackpropFilter(x, out, shape)

    @fns
    def func_dynamic_rnn(x, w, b, seq_length, init_h, init_c):
        return DynamicRNN(x, w, b, seq_length, init_h, init_c)

    @fns
    def func_layer_norm(input_x, gamma, beta):
        return LayerNorm(input_x, gamma, beta)

    @fns
    def fusion_add_conv2d(x, y, z):
        add = Add(x, y)
        return Conv2D(add, z)

    return fns[tag]
