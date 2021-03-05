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

tuple_getitem = Primitive(Constants.kTupleGetItem)
add = P.Add()
max_pool = P.MaxPoolWithArgmax(pad_mode="same", kernel_size=3, strides=2)
make_tuple = Primitive('MakeTuple')
transdata = Primitive("TransData")
Transpose = P.Transpose()


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_transpose_transdata_fusion(tag):
    fns = FnDict()

    @fns
    def before(x):
        res = Transpose(x, (1, 0, 2, 3))
        return res

    @fns
    def after(x):
        output = transdata(x)
        res = make_tuple(output)
        return res

    return fns[tag]
