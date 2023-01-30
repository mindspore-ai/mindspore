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

depend = P.Depend()
all_reduce = P.AllReduce()
broadcast = P.Broadcast(1)
tensor_move = Primitive('TensorMove')
make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)
assign_add = P.AssignAdd()
apply_momentun = P.ApplyMomentum()
relu = P.ReLU()
relu_tbe = Primitive('Relu')


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_insert_tensor_move_for_hccl_op_cond1(tag):
    fns = FnDict()

    @fns
    def before1(x):
        res1 = relu(x)
        res2 = all_reduce(res1)
        return make_tuple(res1, res2)

    @fns
    def before2(x):
        res1 = relu(x)
        res2 = all_reduce(res1)
        return res2

    @fns
    def after(x):
        res1 = relu(x)
        res2 = tensor_move(res1)
        res2 = all_reduce(res2)
        return make_tuple(make_tuple(res1, res2))

    return fns[tag]


def test_insert_tensor_move_for_hccl_op_cond2(tag):
    fns = FnDict()

    @fns
    def before(x):
        res = all_reduce(x)
        return res

    @fns
    def after(x):
        res = tensor_move(x)
        res = all_reduce(res)
        return make_tuple(res)

    return fns[tag]


def test_insert_tensor_move_for_hccl_op_cond3(tag):
    fns = FnDict()

    @fns
    def before(a, b):
        res = assign_add(a, b)
        res = all_reduce(res)
        return res

    @fns
    def after(a, b):
        res = assign_add(a, b)
        res = tensor_move(res)
        res = all_reduce(res)
        return make_tuple(res)

    return fns[tag]


def test_insert_tensor_move_for_hccl_op_cond4(tag):
    fns = FnDict()

    @fns
    def before(a, b):
        x = relu(a)
        y = all_reduce(b)
        res = depend(x, y)
        return res

    @fns
    def after(a, b):
        x = relu_tbe(a)
        y1 = tensor_move(b)
        y2 = all_reduce(y1)
        res = depend(x, y2)
        return make_tuple(res)

    return fns[tag]


def test_insert_tensor_move_for_hccl_op_cond5(tag):
    fns = FnDict()

    @fns
    def before(a, b, c):
        x = relu(a)
        y = broadcast((b, c))
        res = depend(x, y)
        return res

    @fns
    def after(a, b, c):
        x = relu_tbe(a)
        m1 = tensor_move(b)
        m2 = tensor_move(c)
        y = broadcast(m1, m2)
        res = depend(x, y)
        return make_tuple(res)

    return fns[tag]
