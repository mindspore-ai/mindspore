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

tensor_scatter_update = P.TensorScatterUpdate()
tensor_move = Primitive('TensorMove')
scatter_nd_update = Primitive('ScatterNdUpdate')
make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_tensor_scatter_update_fission(tag):
    fns = FnDict()

    @fns
    def before(x, indices, updates):
        res = tensor_scatter_update(x, indices, updates)
        return res

    @fns
    def after(x, indices, updates):
        res = tensor_move(x)
        res = scatter_nd_update(res, indices, updates)
        return make_tuple(res)

    return fns[tag]
