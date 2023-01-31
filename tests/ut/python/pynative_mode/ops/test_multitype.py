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
""" test_multitype """
import numpy as np

from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.common.parameter import Parameter
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import dtype as mstype
from ...ut_filter import non_graph_engine

tensor_add = P.Add()
op_add = P.AddN()
scala_add = F.scalar_add
add = C.MultitypeFuncGraph('add')


@add.register("Number", "Number")
def add_scala(x, y):
    return scala_add(x, y)


@add.register("Tensor", "Tensor")
def add_tensor(x, y):
    return tensor_add(x, y)


@jit
def mainf(x, y):
    return add(x, y)


@non_graph_engine
def test_multitype_tensor():
    tensor1 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    tensor2 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    mainf(tensor1, tensor2)


@non_graph_engine
def test_multitype_tuple():
    tensor1 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    params1 = Parameter(tensor1, name="params1")
    tensor2 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    output = op_add((params1, tensor2))
    assert np.all(output.asnumpy() == np.array([[2.4, 4.2], [4.4, 6.4]]).astype('float32'))


def test_multitype_scalar():
    mainf(1, 2)


add2 = C.MultitypeFuncGraph('add2')
@add2.register(mstype.number, mstype.number)
def add_scala2(x, y):
    return scala_add(x, y)


@add2.register(mstype.tensor, mstype.tensor)
def add_tensor2(x, y):
    return tensor_add(x, y)


@jit
def mainf2(x, y):
    return add2(x, y)


@non_graph_engine
def test_multitype_tensor_by_type():
    tensor1 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    tensor2 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    out = mainf2(tensor1, tensor2)
    print(out)
