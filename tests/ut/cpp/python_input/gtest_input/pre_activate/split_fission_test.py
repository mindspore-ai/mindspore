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

split = P.Split(0, 8)
make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)
splitv = Primitive('SplitV')


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_split_fission(tag):
    """ test_adam_apply_one_with_decay_rule """
    fns = FnDict()

    @fns
    def before(x):
        return split(x)

    @fns
    def after(x):
        splitv0 = splitv(x)
        splitv1 = splitv(tuple_getitem(splitv0, 0))
        splitv2 = splitv(tuple_getitem(splitv0, 1))
        splitv3 = splitv(tuple_getitem(splitv0, 2))
        make_tuple0 = make_tuple(tuple_getitem(splitv1, 0), tuple_getitem(splitv1, 1), tuple_getitem(splitv1, 2),
                                 tuple_getitem(splitv2, 0), tuple_getitem(splitv2, 1), tuple_getitem(splitv2, 2),
                                 tuple_getitem(splitv3, 0), tuple_getitem(splitv3, 1))
        return make_tuple(
            make_tuple(tuple_getitem(make_tuple0, 0), tuple_getitem(make_tuple0, 1), tuple_getitem(make_tuple0, 2),
                       tuple_getitem(make_tuple0, 3), tuple_getitem(make_tuple0, 4), tuple_getitem(make_tuple0, 5),
                       tuple_getitem(make_tuple0, 6), tuple_getitem(make_tuple0, 7)))

    return fns[tag]
