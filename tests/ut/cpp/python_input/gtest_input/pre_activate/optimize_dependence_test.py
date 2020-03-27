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
from mindspore.ops import operations as P
from mindspore.ops import Primitive

depend = Primitive('depend')
TransData = Primitive('TransData')
add = P.TensorAdd()

class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_optimize_dependence(tag):
    fns = FnDict()

    @fns
    def before(x, y, z):
        new_z = TransData(z)
        depend_intput = depend(y, new_z)
        sum = add(x, depend_intput)
        return sum

    @fns
    def after(x, y, z):
        depend_intput = depend(y, z)
        sum = add(x, depend_intput)
        return sum

    return fns[tag]
