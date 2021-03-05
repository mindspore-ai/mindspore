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

maximum = P.Maximum()
minimum = P.Minimum()
clip_by_value = Primitive('ClipByValue')
make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_clip_by_value_fusion(tag):
    """ test_clip_by_value_fusion """
    fns = FnDict()

    @fns
    def before1(input0, input1, input2):
        res = minimum(input0, input1)
        res = maximum(res, input2)
        return res

    @fns
    def before2(input0, input1, input2):
        res = minimum(input0, input1)
        res = maximum(input2, res)
        return res

    @fns
    def after(input0, input1, input2):
        res = clip_by_value(input0, input2, input1)
        return make_tuple(res)

    return fns[tag]
