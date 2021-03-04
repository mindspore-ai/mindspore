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

relu = P.ReLU()
relu_grad = Primitive('ReluGrad')
relu_v2 = Primitive('ReLUV2')
relu_grad_v2 = Primitive('ReluGradV2')
make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_derelu_fusion(tag):
    fns = FnDict()

    @fns
    def before(i0, i1):
        relu_res = relu(i1)
        res = relu_grad(i0, relu_res)
        other = relu(relu_res)
        res = make_tuple(res, other)
        return res

    @fns
    def after(i0, i1):
        relu_res = relu_v2(i1)
        item0 = tuple_getitem(relu_res, 0)
        item1 = tuple_getitem(relu_res, 1)
        other = relu(item0)
        res = relu_grad_v2(i0, item1)
        res = make_tuple(res, other)
        return make_tuple(res)

    return fns[tag]
