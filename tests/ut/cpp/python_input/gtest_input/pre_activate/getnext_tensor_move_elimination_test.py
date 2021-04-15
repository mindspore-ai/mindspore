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
from mindspore.ops import operations as P

get_next = P.GetNext([ms.float32], [[1, 64, 112, 112]], 1, "")
tensor_move_attr = Primitive('TensorMove')
tensor_move_attr.add_prim_attr("label_for_insert_stream_active", True)
tensor_move = Primitive('tensor_move')
cast = P.Cast()
add = P.Add()


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_getnext_tensor_move_elimination(tag):
    fns = FnDict()

    @fns
    def before():
        res = get_next()
        res = tensor_move_attr(res)
        res = cast(res)
        res = add(res)
        return res

    @fns
    def after():
        res = get_next()
        res = cast(res)
        res = add(res)
        return res

    return fns[tag]


def test_getnext_tensor_move_elimination_no_attr(tag):
    fns = FnDict()

    @fns
    def before():
        res = get_next()
        res = tensor_move(res)
        res = cast(res)
        return res

    @fns
    def after():
        res = get_next()
        res = tensor_move(res)
        res = cast(res)
        return res

    return fns[tag]


def test_getnext_tensor_move_elimination_tensor_move_multi_users(tag):
    fns = FnDict()

    @fns
    def before():
        res = get_next()
        tensor_move_out = tensor_move_attr(res)
        res = cast(tensor_move_out)
        res = add(tensor_move_out, res)
        return res

    @fns
    def after():
        res = get_next()
        tensor_move_out = tensor_move_attr(res)
        res = cast(tensor_move_out)
        res = add(tensor_move_out, res)
        return res

    return fns[tag]


def test_getnext_tensor_move_elimination_next_multi_inputs(tag):
    fns = FnDict()

    @fns
    def before():
        res = get_next()
        tensormove_out = tensor_move_attr(res)
        res = add(tensormove_out, res)
        return res

    @fns
    def after():
        res = get_next()
        tensormove_out = tensor_move_attr(res)
        res = add(tensormove_out, res)
        return res

    return fns[tag]
