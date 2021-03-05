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

tuple_getitem = Primitive(Constants.kTupleGetItem)
add = P.Add()
max_pool = P.MaxPoolWithArgmax(pad_mode="same", kernel_size=3, strides=2)
make_tuple = Primitive('MakeTuple')
trans_data = Primitive("TransData")


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_remove_internal_output_trans_op_for_single_output(tag):
    fns = FnDict()

    @fns
    def before(x, y):
        res = add(x, y)
        return res

    @fns
    def after_insert_trans_op(x, y):
        output = add(x, y)
        res = trans_data(output)
        return make_tuple(res)

    @fns
    def after_remove_internal_output_trans_op(x, y):
        res = add(x, y)
        return make_tuple(res)

    return fns[tag]


def test_remove_internal_output_trans_op_for_multiple_output(tag):
    fns = FnDict()

    @fns
    def before(x):
        max_pool_res = max_pool(x)
        res = make_tuple(tuple_getitem(max_pool_res, 0), tuple_getitem(max_pool_res, 1))
        return res

    @fns
    def after_insert_trans_op(x):
        output = max_pool(x)
        trans_data0 = trans_data(tuple_getitem(output, 0))
        trans_data1 = trans_data(tuple_getitem(output, 1))
        new_make_tuple = make_tuple(trans_data0, trans_data1)
        res = make_tuple(tuple_getitem(new_make_tuple, 0), tuple_getitem(new_make_tuple, 1))
        return make_tuple(res)

    @fns
    def after_remove_internal_output_trans_op(x):
        output = max_pool(x)
        new_make_tuple = make_tuple(tuple_getitem(output, 0), tuple_getitem(output, 1))
        res = make_tuple(tuple_getitem(new_make_tuple, 0), tuple_getitem(new_make_tuple, 1))
        return make_tuple(res)

    return fns[tag]
