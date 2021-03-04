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
import numpy as np

from mindspore.common.tensor import Tensor
from mindspore.ops import Primitive
from mindspore.ops import operations as P

make_tuple = Primitive('MakeTuple')
concat = P.Concat()
add = P.Add()

t1 = Tensor(np.random.randn(1, 11, 20, 1, 1).astype(np.float32))
t2 = Tensor(np.random.randn(1, 11, 20, 1, 1).astype(np.float32))


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_convert_tuple_input_to_dynamic_input(tag):
    fns = FnDict()

    @fns
    def before(x):
        res = concat((x, x))
        res = add(x, res)
        return res

    @fns
    def after(x):
        res = concat(x, x)
        res = add(x, res)
        res = make_tuple(res)
        return res

    return fns[tag]
