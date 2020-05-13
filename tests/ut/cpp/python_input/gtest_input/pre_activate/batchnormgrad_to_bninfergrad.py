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
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops import Primitive

batch_norm_grad = G.BatchNormGrad(is_training=False)
bn_infer_grad = Primitive('BNInferGrad')
make_tuple = Primitive('make_tuple')
tuple_getitem = Primitive('tuple_getitem')

class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]

def test_batchnormgrad_to_bninfergrad(tag):
    fns = FnDict()

    @fns
    def before(input0, input1, input2, input3, input4):
        res = batch_norm_grad(input0, input1, input2, input3, input4)
        res = tuple_getitem(res, 0)
        return res

    @fns
    def after(input0, input1, input2, input3, input4):
        res = bn_infer_grad(input0, input2, input4)
        return make_tuple(res)

    @fns
    def no_fusion(input0, input1, input2, input3, input4):
        res = batch_norm_grad(input0, input1, input2, input3, input4)
        item0 = tuple_getitem(res, 0)
        item1 = tuple_getitem(res, 1)
        item2 = tuple_getitem(res, 2)
        return make_tuple(item0, item1, item2)

    return fns[tag]
