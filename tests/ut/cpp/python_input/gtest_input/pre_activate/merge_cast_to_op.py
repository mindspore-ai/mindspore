# Copyright 2019 Huawei Technologies Co., Ltd
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

make_tuple = Primitive('MakeTuple')
four2five = Primitive('Four2Five')
five2four = Primitive('Five2Four')
cast = Primitive('Cast')


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_merge_cast_to_next_op(tag):
    fns = FnDict()

    @fns
    def before(x):
        x = cast(x)
        output = four2five(x)
        return output

    @fns
    def after(x):
        output = four2five(x)
        return make_tuple(output)

    return fns[tag]


def test_merge_cast_to_prior_op(tag):
    fns = FnDict()

    @fns
    def before(x):
        x = five2four(x)
        output = cast(x)
        return output

    @fns
    def after(x):
        output = five2four(x)
        return make_tuple(output)

    return fns[tag]
