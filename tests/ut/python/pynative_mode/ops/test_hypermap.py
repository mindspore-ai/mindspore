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
""" test_hypermap """
import numpy as np

from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from ...ut_filter import non_graph_engine

# pylint: disable=W0613
# W0613: unused-argument


tensor_add = P.Add()
scala_add = F.scalar_add
add = C.MultitypeFuncGraph('add')


@add.register("Number", "Number")
def add_scala(x, y):
    return scala_add(x, y)


@add.register("Tensor", "Tensor")
def add_tensor(x, y):
    return tensor_add(x, y)


hyper_add = C.HyperMap(add)


@jit
def mainf(x, y):
    return hyper_add(x, y)


@non_graph_engine
def test_hypermap_tensor():
    tensor1 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    tensor2 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    print("test_hypermap_tensor:", mainf(tensor1, tensor2))


def test_hypermap_scalar():
    print("test_hypermap_scalar", mainf(1, 2))


def test_hypermap_tuple():
    print("test_hypermap_tuple", mainf((1, 1), (2, 2)))


@non_graph_engine
def test_hypermap_tuple_tensor():
    tensor1 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    tensor2 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    print("test_hypermap_tuple_tensor", mainf((tensor1, tensor1), (tensor2, tensor2)))


@non_graph_engine
def test_hypermap_tuple_mix():
    tensor1 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    tensor2 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    print("test_hypermap_tuple_mix", mainf((tensor1, 1), (tensor2, 2)))


hyper_map = C.HyperMap()


@jit
def main_noleaf(x, y):
    return hyper_map(add, x, y)


def test_hypermap_noleaf_scalar():
    main_noleaf(1, 2)


@non_graph_engine
def test_hypermap_noleaf_tensor():
    tensor1 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    tensor2 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    main_noleaf(tensor1, tensor2)


def test_hypermap_noleaf_tuple():
    main_noleaf((1, 1), (2, 2))


@non_graph_engine
def test_hypermap_noleaf_tuple_tensor():
    tensor1 = Tensor(np.array([[1.1, 2.1], [2.1, 3.1]]).astype('float32'))
    tensor2 = Tensor(np.array([[1.2, 2.2], [2.2, 3.2]]).astype('float32'))
    tensor3 = Tensor(np.array([[2.2], [3.2]]).astype('float32'))
    tensor4 = Tensor(np.array([[2.2], [3.2]]).astype('float32'))
    main_noleaf((tensor1, tensor3), (tensor2, tensor4))


def test_hypermap_noleaf_tuple_mix():
    tensor1 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    tensor2 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    main_noleaf((tensor1, 1), (tensor2, 2))


def add3_scalar(x, y, z):
    return scala_add(scala_add(x, y), z)


@jit
def main_add3_easy(x, y):
    add2 = F.partial(add3_scalar, 1)
    return add2(x, y)


def test_hypermap_add3_easy():
    main_add3_easy(1, 2)


add3 = C.MultitypeFuncGraph('add')
partial = P.Partial()


@add3.register("Number", "Number", "Number")
def add3_scala(x, y, z):
    return scala_add(scala_add(x, y), z)


@add3.register("Number", "Tensor", "Tensor")
def add3_tensor(x, y, z):
    return tensor_add(y, z)


@jit
def main_add3_scala(x, y):
    add2 = partial(add3_scala, 1)
    return hyper_map(add2, x, y)


@jit
def main_add3(x, y):
    add2 = partial(add3, 1)
    return hyper_map(add2, x, y)


@non_graph_engine
def test_hypermap_add3_tensor():
    tensor1 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    tensor2 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    main_add3(tensor1, tensor2)


def test_hypermap_add3_tuple():
    tensor1 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    tensor2 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))

    main_add3((tensor1, 1), (tensor2, 1))
