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

select = P.Select()
maximum = P.Maximum()
sqrt = P.Sqrt()
greater = P.Greater()
clip_by_norm_no_div_square_sum = Primitive('ClipByNormNoDivSum')
make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_clip_by_norm_no_div_square_sum_fusion(tag):
    """ test_clip_by_norm_no_div_square_sum_fusion """
    fns = FnDict()

    @fns
    def before(x, constant_select, constant_greater, constant_maximum):
        greater_output = greater(x, constant_greater)
        res = select(greater_output, x, constant_select)
        res = sqrt(res)
        res = select(greater_output, res, x)
        res = maximum(res, constant_maximum)
        return res

    @fns
    def after(x, constant_select, constant_greater, constant_maximum):
        res = clip_by_norm_no_div_square_sum(x, constant_select, constant_greater, constant_maximum)
        return make_tuple(res)

    return fns[tag]
