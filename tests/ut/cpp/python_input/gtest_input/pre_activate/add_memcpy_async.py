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
from mindspore.ops import Primitive

all_reduce = P.AllReduce()
memcpy_async = Primitive('memcpy_async')
make_tuple = Primitive('make_tuple')
tuple_getitem = Primitive('tuple_getitem')


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_add_memcpy_async(tag):
    fns = FnDict()

    @fns
    def before(x):
        res = all_reduce(x)
        return make_tuple(x, res)

    @fns
    def after(x):
        res = memcpy_async(x)
        res = all_reduce(res)
        return make_tuple(make_tuple(x, res))

    return fns[tag]
