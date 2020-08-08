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

concat = P.Concat()


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_concat_fission(tag):
    """ test_adam_apply_one_with_decay_rule """
    fns = FnDict()

    @fns
    def before(input0, input1, input2, input3, input4, input5, input6, input7, input8):
        return concat((input0, input1, input2, input3, input4, input5, input6, input7, input8))

    @fns
    def after_divided_by_2(input0, input1, input2, input3, input4, input5, input6, input7, input8):
        a = concat((input0, input1))
        b = concat((input2, input3))
        c = concat((input4, input5))
        d = concat((input6, input7))
        f = concat((a, b))
        g = concat((c, d))
        i = concat((f, g))
        return concat((i, input8))

    @fns
    def after_divided_by_3(input0, input1, input2, input3, input4, input5, input6, input7, input8):
        a = concat((input0, input1, input2))
        b = concat((input3, input4, input5))
        c = concat((input6, input7, input8))
        return concat((a, b, c))

    @fns
    def after_divided_by_4(input0, input1, input2, input3, input4, input5, input6, input7, input8):
        a = concat((input0, input1, input2, input3))
        b = concat((input4, input5, input6, input7))
        return concat((a, b, input8))

    @fns
    def after_divided_by_8(input0, input1, input2, input3, input4, input5, input6, input7, input8):
        a = concat((input0, input1, input2, input3, input4, input5, input6, input7))
        return concat((a, input8))

    @fns
    def after_divided_by_9(input0, input1, input2, input3, input4, input5, input6, input7, input8):
        return concat((input0, input1, input2, input3, input4, input5, input6, input7, input8))

    return fns[tag]
