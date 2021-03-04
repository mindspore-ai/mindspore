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

Transpose = P.Transpose()
Reshape = P.Reshape()
ConfusionTransposeD = Primitive('ConfusionTransposeD')
make_tuple = Primitive('MakeTuple')


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_reshape_transpose_fusion(tag):
    fns = FnDict()

    @fns
    def before(input0):
        reshape = Reshape(input0, (2, 2, 16, 16))
        transpose = Transpose(reshape, (1, 0, 2, 3))
        return transpose

    @fns
    def after(input0):
        confusion = ConfusionTransposeD(input0)
        res = make_tuple(confusion)
        return res

    return fns[tag]
