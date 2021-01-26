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

Add = P.Add()
Cast = P.Cast()
LayerNormBetaGammaBackprop = Primitive('LayerNormBetaGammaBackprop')
tuple_getitem = Primitive(Constants.kTupleGetItem)


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_layer_norm_beta_gamma_backprop_fusion(tag):
    fns = FnDict()

    @fns
    def before(input0, input1, input2, input3):
        layer = LayerNormBetaGammaBackprop(input0, input1, input2, input3)
        output0 = Cast(tuple_getitem(layer, 0))
        output1 = Cast(tuple_getitem(layer, 1))
        add = Add(output0, output1)
        return add

    @fns
    def before_unmatched_inputs_size(input0, input1, input2):
        layer = LayerNormBetaGammaBackprop(input0, input1, input2)
        output0 = Cast(tuple_getitem(layer, 0))
        output1 = Cast(tuple_getitem(layer, 1))
        add = Add(output0, output1)
        return add

    @fns
    def before_unmatched_outputs_size(input0, input1, input2, input3):
        layer = LayerNormBetaGammaBackprop(input0, input1, input2, input3)
        output0 = Cast(layer)
        return output0

    @fns
    def after(input0, input1, input2, input3):
        layer = LayerNormBetaGammaBackprop(input0, input1, input2, input3)
        output0 = tuple_getitem(layer, 0)
        output1 = tuple_getitem(layer, 1)
        add = Add(output0, output1)
        return add

    return fns[tag]
