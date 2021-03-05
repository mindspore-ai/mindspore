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
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops import _constants as Constants

make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)
layer_norm_grad = G.LayerNormGrad()
layer_norm_x_backprop = Primitive('LayerNormXBackprop')
layer_norm_beta_gamma_backprop = Primitive('LayerNormBetaGammaBackprop')


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_layer_norm_grad_split(tag):
    """ test_layer_norm_grad_split """
    fns = FnDict()

    @fns
    def before(i0, i1, i2, i3, i4):
        layer_norm_grad_output = layer_norm_grad(i0, i1, i2, i3, i4)
        item0 = tuple_getitem(layer_norm_grad_output, 0)
        item1 = tuple_getitem(layer_norm_grad_output, 1)
        item2 = tuple_getitem(layer_norm_grad_output, 2)
        res = make_tuple(item0, item1, item2)
        return res

    @fns
    def after(i0, i1, i2, i3, i4):
        layer_norm_x_output = layer_norm_x_backprop(i0, i1, i2, i3, i4)
        layer_norm_beta_output = layer_norm_beta_gamma_backprop(i0, i1, i2, i3)
        beta_item0 = tuple_getitem(layer_norm_beta_output, 0)
        beta_item1 = tuple_getitem(layer_norm_beta_output, 1)
        mt = make_tuple(layer_norm_x_output, beta_item0, beta_item1)
        item0 = tuple_getitem(mt, 0)
        item1 = tuple_getitem(mt, 1)
        item2 = tuple_getitem(mt, 2)
        res = make_tuple(item0, item1, item2)
        return make_tuple(res)

    return fns[tag]
