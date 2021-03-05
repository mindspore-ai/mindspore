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

import mindspore as ms
from mindspore.ops import Primitive
from mindspore.ops import _constants as Constants
from mindspore.ops import operations as P

get_next = P.GetNext([ms.float32, ms.int32], [[32, 64], [32]], 2, "")
memcpy_async = Primitive('memcpy_async')
make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_insert_memcpy_async_for_getnext(tag):
    fns = FnDict()

    @fns
    def getnext_multi_output_before():
        res = get_next()
        return res

    @fns
    def getnext_multi_output_after():
        res = get_next()
        data = tuple_getitem(res, 0)
        label = tuple_getitem(res, 1)
        memcpy_async_data = memcpy_async(data)
        memcpy_async_label = memcpy_async(label)
        bind_tuple = make_tuple(memcpy_async_data, memcpy_async_label)
        get_item0 = tuple_getitem(bind_tuple, 0)
        get_item1 = tuple_getitem(bind_tuple, 1)
        bind_tuple = make_tuple(make_tuple(get_item0, get_item1))
        return bind_tuple

    return fns[tag]
