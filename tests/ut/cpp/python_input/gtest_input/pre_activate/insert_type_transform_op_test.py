# Copyright 2022 Huawei Technologies Co., Ltd
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
import mindspore
from mindspore.common.tensor import Tensor
from mindspore.ops import Primitive
from mindspore.ops import operations as P
from mindspore.ops import _constants as Constants
from mindspore.ops.operations import _sequence_ops as seq

tuple_get_item = Primitive(Constants.kTupleGetItem)

make_tuple = Primitive('MakeTuple')
split1 = P.Split(1, 2)
split2 = P.Split(0, 2)
add1 = P.AddN()
add2 = P.AddN()

input_x = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]), mindspore.int32)


class FnDict:
    def __init__(self):
        self.fn_dict = {}

    def __call__(self, fn):
        self.fn_dict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fn_dict.get(name)


def test_tuple_unfold_to_tuple_unfold_transform(tag):
    """
    Feature: Dynamic shape.
    Description: Test TupleUnfold to TupleUnfold transforming pass.
    Expectation: The 'after' graph is identical to the graph after this pass.
    """
    fns = FnDict()

    @fns
    def before(x):
        res = split1(x)
        res = add1(res)
        res = split2(res)
        res = add2((tuple_get_item(res, 0), tuple_get_item(res, 1)))
        return res

    @fns
    def after(x):
        res = split1(x)
        res = add1(tuple_get_item(res, 0), tuple_get_item(res, 1))
        res = split2(res)
        res = add2(tuple_get_item(res, 0), tuple_get_item(res, 1))
        return res

    return fns[tag]


def test_tuple_unfold_to_tuple_transform(tag):
    """
    Feature: Dynamic shape.
    Description: Test TupleUnfold to Tuple transforming pass.
    Expectation: The 'after' graph is identical to the graph after this pass.
    """
    fns = FnDict()
    # Need to change AddN to SequenceAdd in later version. This case is just used to cover this pattern.
    seq_add1 = seq.SequenceAdd()
    tensor_to_scalar = Primitive('TensorToScalar')
    real_make_tuple = Primitive('RealMakeTuple')

    @fns
    def before(input_1, input_2, x):
        res = make_tuple(input_1, input_2)
        res = seq_add1(res, x)
        return res

    @fns
    def after(input_1, input_2, x):
        input_1 = tensor_to_scalar(input_1)
        input_2 = tensor_to_scalar(input_2)
        res = real_make_tuple(input_1, input_2)
        res = seq_add1(res, x)
        return res

    return fns[tag]


def test_tuple_unfold_to_tensor_transform(tag):
    """
    Feature: Dynamic shape.
    Description: Test TupleUnfold to Tensor transforming pass.
    Expectation: The 'after' graph is identical to the graph after this pass.
    """
    fns = FnDict()
    reshape = P.Reshape()
    real_make_tuple = Primitive('RealMakeTuple')
    tuple_to_tensor = Primitive('TupleToTensor')
    tensor_to_scalar = Primitive('TensorToScalar')

    @fns
    def before(input_1, input_2, input_3):
        res = make_tuple(input_1, input_2)
        res = reshape(input_3, res)
        return res

    @fns
    def after(input_1, input_2, input_3):
        input_1 = tensor_to_scalar(input_1)
        input_2 = tensor_to_scalar(input_2)
        res = real_make_tuple(input_1, input_2)
        res = tuple_to_tensor(res)
        res = reshape(input_3, res)
        return res

    return fns[tag]


def test_tuple_to_tuple_unfold_transform(tag):
    """
    Feature: Dynamic shape.
    Description: Test Tuple to TupleUnfold transforming pass.
    Expectation: The 'after' graph is identical to the graph after this pass.
    """
    fns = FnDict()
    shape = P.Shape()
    real_tuple_get_item = Primitive('RealTupleGetItem')

    @fns
    def before(x):
        res = shape(x)
        res = res[0]
        return res

    @fns
    def after(x):
        res = shape(x, y)
        res = real_tuple_get_item(res, 0)
        return res

    return fns[tag]


def test_tuple_to_tensor_transform(tag):
    """
    Feature: Dynamic shape.
    Description: Test Tuple to Tensor transforming pass.
    Expectation: The 'after' graph is identical to the graph after this pass.
    """
    fns = FnDict()
    reshape = P.Reshape()
    tuple_to_tensor = Primitive('TupleToTensor')

    @fns
    def before(x, y):
        res = reshape(x, y)
        return res

    @fns
    def after(x, y):
        res = tuple_to_tensor(y)
        res = reshape(x, res)
        return res

    return fns[tag]


def test_scalar_to_tensor_transform(tag):
    """
    Feature: Dynamic shape.
    Description: Test Scalar to Tensor transforming pass.
    Expectation: The 'after' graph is identical to the graph after this pass.
    """
    fns = FnDict()
    add = P.Add()
    scalar_to_tensor = Primitive('ScalarToTensor')

    @fns
    def before(x, y):
        res = add(x, y)
        return res

    @fns
    def after(x, y):
        x = scalar_to_tensor(x)
        y = scalar_to_tensor(y)
        res = add(x, y)
        return res

    return fns[tag]


def test_tensor_to_tuple_transform(tag):
    """
    Feature: Dynamic shape.
    Description: Test Tensor to Tuple transforming pass.
    Expectation: The 'after' graph is identical to the graph after this pass.
    """
    fns = FnDict()
    # Need to change Add to SequenceAdd in later version. This case is just used to cover this pattern.
    seq_add = P.Add()
    tensor_to_tuple = Primitive('TensorToTuple')

    @fns
    def before(x, y):
        res = seq_add(x, y)
        return res

    @fns
    def after(x, y):
        input1 = tensor_to_tuple(x)
        input2 = tensor_to_tuple(y)
        res = seq_add(input1, input2)
        return res

    return fns[tag]


def test_tensor_to_scalar_transform(tag):
    """
    Feature: Dynamic shape.
    Description: Test Tensor to Scalar transforming pass.
    Expectation: The 'after' graph is identical to the graph after this pass.
    """
    fns = FnDict()
    scalar_to_tensor = P.ScalarToTensor()
    tensor_to_scalar = Primitive('TensorToScalar')

    @fns
    def before(x):
        res = scalar_to_tensor(x)
        return res

    @fns
    def after(x):
        res = tensor_to_scalar(x)
        res = scalar_to_tensor(res)
        return res

    return fns[tag]
