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

addn = P.AddN()
make_tuple = Primitive('MakeTuple')


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_addn_fission(tag):
    """ test_adam_apply_one_with_decay_rule """
    fns = FnDict()

    @fns
    def before(input0, input1, input2, input3, input4, input5, input6, input7, input8):
        return addn((input0, input1, input2, input3, input4, input5, input6, input7, input8))

    @fns
    def after_divided_by_2(input0, input1, input2, input3, input4, input5, input6, input7, input8):
        a = addn((input0, input1))
        b = addn((input2, input3))
        c = addn((input4, input5))
        d = addn((input6, input7))
        f = addn((a, b))
        g = addn((c, d))
        i = addn((f, g))
        return addn((i, input8))

    @fns
    def after_divided_by_3(input0, input1, input2, input3, input4, input5, input6, input7, input8):
        a = addn((input0, input1, input2))
        b = addn((input3, input4, input5))
        c = addn((input6, input7, input8))
        return addn((a, b, c))

    @fns
    def after_divided_by_4(input0, input1, input2, input3, input4, input5, input6, input7, input8):
        a = addn((input0, input1, input2, input3))
        b = addn((input4, input5, input6, input7))
        return addn((a, b, input8))

    @fns
    def after_divided_by_8(input0, input1, input2, input3, input4, input5, input6, input7, input8):
        a = addn((input0, input1, input2, input3, input4, input5, input6, input7))
        return addn((a, input8))

    @fns
    def after_divided_by_9(input0, input1, input2, input3, input4, input5, input6, input7, input8):
        return addn((input0, input1, input2, input3, input4, input5, input6, input7, input8))

    return fns[tag]
