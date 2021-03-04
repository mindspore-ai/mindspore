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
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import Primitive
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G

make_tuple = Primitive('MakeTuple')
reshape = P.Reshape()
backend_reshape = Primitive('Reshape')
cast = P.Cast()
backend_cast = Primitive('Cast')
transpose = P.Transpose()
backend_transpose = Primitive('Transpose')
onehot1 = P.OneHot()
onehot2 = P.OneHot()
backend_onehot1 = Primitive('OneHot')
backend_onehot2 = Primitive('OneHot')
stridedslicegrad = G.StridedSliceGrad()
backend_stridedslicegrad = Primitive('StridedSliceGrad')
on_value = Tensor(1.0, mstype.float32)
off_value = Tensor(0.0, mstype.float32)
depth = Tensor(2, mstype.int32)
shape = (2, 4, 2, 2)
dropout_gen_mask = P.DropoutGenMask()


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_convert_reshape_input_to_attr(tag):
    fns = FnDict()

    @fns
    def before(x):
        return reshape(x, (3, 2))

    @fns
    def after(x):
        res = backend_reshape(x)
        return make_tuple(res)

    return fns[tag]


def test_convert_cast_input_to_attr(tag):
    fns = FnDict()

    @fns
    def before(x):
        return cast(x, ms.int32)

    @fns
    def after(x):
        res = backend_cast(x)
        return make_tuple(res)

    return fns[tag]


def test_convert_transpose_input_to_attr(tag):
    fns = FnDict()

    @fns
    def before(x):
        return transpose(x, (0, 2, 1))

    @fns
    def after(x):
        res = backend_transpose(x)
        return make_tuple(res)

    return fns[tag]


def test_convert_onehot_input_to_attr(tag):
    fns = FnDict()

    @fns
    def before(x):
        return onehot1(x, 2, on_value, off_value)

    @fns
    def after(x):
        res = backend_onehot1(x, on_value, off_value)
        return make_tuple(res)

    return fns[tag]


def test_convert_onehot_input_to_tensor1(tag):
    fns = FnDict()

    @fns
    def before(x):
        return onehot2(x, 2, on_value, off_value)

    @fns
    def after_func_graph(x):
        res = backend_onehot2(x, depth, on_value, off_value)
        return make_tuple(res)

    return fns[tag]


def test_convert_onehot_input_to_tensor2(tag):
    fns = FnDict()

    @fns
    def before(x):
        return onehot2(x, 2, on_value, off_value)

    @fns
    def after_kernel_graph(x):
        return make_tuple(backend_onehot2(x, on_value, off_value))

    return fns[tag]


def test_convert_dropout_gen_mask_tuple_input_to_tensor(tag):
    fns = FnDict()

    @fns
    def before(x):
        return dropout_gen_mask(shape, x)

    return fns[tag]
