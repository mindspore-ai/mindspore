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
""" test_tensor_slice """
import numpy as np
import pytest
import torch
import torch.nn as nn_torch

from mindspore import nn
from mindspore import Tensor, Parameter
from mindspore import context, jit, ops
from mindspore import dtype as mstype
from mindspore.nn import Cell
from mindspore.graph.api import _pynative_executor

from tests.mark_utils import arg_mark
from tests.mindspore_test_framework.mindspore_test import mindspore_test
from tests.mindspore_test_framework.pipeline.forward.compile_forward \
    import pipeline_for_compile_forward_ge_graph_for_case_by_case_config, \
    pipeline_for_compile_forward_ge_graph_for_case_by_case_config_exception


class NetWorkSlicePositive(Cell):
    def __init__(self):
        super().__init__()
        self.tensor_ret0 = Tensor(np.ones([1, 2, 2], np.int32))
        self.tensor_ret1 = Tensor(np.ones([4, 7, 4], np.int32))
        self.tensor_ret2 = Tensor(np.ones([6, 8, 10], np.int32))
        self.tensor_ret3 = Tensor(np.ones([3, 8, 10], np.int32))

    def construct(self, tensor):
        ret0 = tensor[3:4:3, 1:5:2, 3:6:2] + self.tensor_ret0
        ret1 = tensor[-6:4:1, 7:-8:-1, ::3] + self.tensor_ret1
        ret2 = tensor[::, ::, ::] + self.tensor_ret2
        ret3 = tensor[::2] + self.tensor_ret3
        return ret0, ret1, ret2, ret3


class NetWorkSliceEllipsis(Cell):
    def __init__(self):
        super().__init__()
        self.tensor_ret0 = Tensor(np.ones([2, 7, 8], np.int32))
        self.tensor_ret1 = Tensor(np.ones([6, 7, 8, 9], np.int32))
        self.tensor_ret2 = Tensor(np.ones([1, 6, 7, 8, 9], np.int32))

    def construct(self, tensor):
        ret0 = tensor[0:4:2, ..., 1] + self.tensor_ret0
        ret1 = tensor[...] + self.tensor_ret1
        ret2 = tensor[None] + self.tensor_ret2
        ret3 = tensor[True] + self.tensor_ret2
        return ret0, ret1, ret2, ret3


class NetWorkReduceDimension(Cell):
    def __init__(self):
        super().__init__()
        self.tensor_ret0 = Tensor(np.ones([2, 4, 1], np.int32))
        self.tensor_ret1 = Tensor(np.ones([3, 4], np.int32))
        self.tensor_ret2 = Tensor(np.ones([6, 8], np.int32))
        self.tensor_ret3 = Tensor(np.array(8, np.int32))
        self.tensor_ret4 = Tensor(np.ones([8, 10], np.int32))

    def construct(self, tensor):
        ret0 = tensor[0:6:3, 1:5:1, 3:5:2] + self.tensor_ret0
        ret1 = tensor[::2, 1, ::3] + self.tensor_ret1
        ret2 = tensor[::, ::, 0] + self.tensor_ret2
        ret3 = tensor[3, 2, 5] + self.tensor_ret3
        ret4 = tensor[1] + self.tensor_ret4
        return ret0, ret1, ret2, ret3, ret4


class NetWorkStepNegative(Cell):
    def __init__(self):
        super().__init__()
        self.tensor_ret = Tensor(np.ones([6, 5, 10], np.int32))

    def construct(self, tensor):
        ret = tensor[::1, -5::, ::-1] + self.tensor_ret
        return ret


class NetWorkReduceToScalar(Cell):
    def __init__(self):
        super().__init__()
        self.tensor_ret = Tensor(np.array(9, np.int32))

    def construct(self, tensor):
        ret = tensor[2, 3, 4] + self.tensor_ret
        return ret


class TensorAssignWithSliceError1(Cell):
    def construct(self, a, b):
        a[1:3:-1, ::] = b
        return a


class TensorAssignWithSliceError2(Cell):
    def construct(self, a, b):
        a[1:3:-1] = b
        return a


class TensorAssignWithSlice2(Cell):
    def construct(self, a, b, ck):
        a[1:5] = b
        a[3:4] = 5
        a[-1:1:-1] = b
        a[-1:3:-1] = 5
        a[::] = b
        a[::] = 9
        z = a + ck
        return z


class TensorAssignWithSlice(Cell):
    def __init__(self):
        super().__init__()
        self.c = 2.0

    def construct(self, a, b, ck):
        a[1:3, ::] = b
        a[2:3:, 3:] = b
        a[::] = b
        a[::] = self.c
        a[::, ::] = b
        a[::, ::] = self.c
        a[2:3:, 0:, 4:1:-1] = b
        a[2:3:, 0:, 4:1:-1] = self.c
        z = a + ck
        return z


class TensorGetItemByOneTensor(Cell):
    def __init__(self):
        super().__init__()
        self.const = Tensor(np.ones((5, 4, 7, 8)), mstype.int32)

    def construct(self, x, index):
        ret = x[index] + self.const
        return ret


class TensorGetItemByTwoTensors(Cell):
    def __init__(self):
        super().__init__()
        self.const = Tensor(np.ones((3, 4, 5, 8)), mstype.int32)

    def construct(self, x, index_0, index_1):
        ret = x[index_0, index_1] + self.const
        return ret


class TensorGetItemByThreeTensors(Cell):
    def __init__(self):
        super().__init__()
        self.const = Tensor(np.ones((5, 3, 4, 5)), mstype.int32)

    def construct(self, x, index_0, index_1, index_2):
        ret = x[index_0, index_1, index_2] + self.const
        return ret


class TensorGetItemByMixedTensors_0(Cell):
    def __init__(self):
        super().__init__()
        self.const = Tensor(np.ones((3, 4, 5, 3, 6, 5), np.float32))

    def construct(self, tensor, index_0, index_1):
        ret = tensor[index_0, index_1, 0:3, ..., 0:5, 3] + self.const
        return ret


class TensorGetItemByMixedTensors_1(Cell):
    def __init__(self):
        super().__init__()
        self.const = Tensor(np.ones((3, 4, 5, 3, 5, 5), np.float32))

    def construct(self, tensor, index_0, index_1):
        ret = tensor[0:3, index_0, ..., index_1, 3, 0:5] + self.const
        return ret


class TensorGetItemByMixedTensors_2(Cell):
    def __init__(self):
        super().__init__()
        self.const = Tensor(np.ones((3, 4, 5, 6, 7), np.float32))

    def construct(self, tensor, index_0, index_1):
        ret = tensor[0, index_0, index_1, ..., 3] + self.const
        return ret


class TensorGetItemByMixedTensors_3(Cell):
    def __init__(self):
        super().__init__()
        self.const = Tensor(np.ones((3, 4, 5, 3, 4, 3, 5), np.float32))

    def construct(self, tensor, index_0, index_1):
        ret = tensor[..., index_0, 0:3, index_1, 0:5] + self.const
        return ret


class TensorGetItemByMixedTensors_4(Cell):
    def __init__(self):
        super().__init__()
        self.const = Tensor(np.ones((2, 2, 3, 4, 5, 3, 9), np.float32))

    def construct(self, tensor, index_0, index_1, index_2):
        ret = tensor[0:2, index_0, index_1, 2, index_2, 0:3, ...] + self.const
        return ret


class TensorGetItemByMixedTensors_5(Cell):
    def __init__(self):
        super().__init__()
        self.const = Tensor(np.ones((2, 3, 4, 5, 2, 6), np.float32))

    def construct(self, tensor, index_0, index_1, index_2):
        ret = tensor[0:2, index_0, index_1, ..., index_2, 2] + self.const
        return ret


class TensorGetItemByMixedTensors_6(Cell):
    def __init__(self):
        super().__init__()
        self.const = Tensor(np.ones((3, 4, 2, 3, 4, 5), np.float32))

    def construct(self, tensor, index_0, index_1, index_2):
        ret = tensor[..., index_0, index_1, index_2, 3] + self.const
        return ret


class TensorSetItemByMixedTensors_0(Cell):
    def __init__(self, value):
        super().__init__()
        self.const = Tensor(np.ones((3, 4, 5, 6, 7, 8, 9), np.float32))
        self.param = Parameter(Tensor(np.arange(3 * 4 * 5 * 6 * 7 * 8 * 9).reshape((3, 4, 5, 6, 7, 8, 9)),
                                      mstype.float32),
                               name="x")
        self.value = value

    def construct(self, index_0, index_1, index_2):
        self.param[0:2, index_0, index_1, 2, index_2, 0:3, ...] = self.value
        ret = self.param + self.const
        return ret


class TensorSetItemByMixedTensors_1(Cell):
    def __init__(self, value):
        super().__init__()
        self.const = Tensor(np.ones((3, 4, 5, 6, 7, 8), np.float32))
        self.param = Parameter(Tensor(np.arange(3 * 4 * 5 * 6 * 7 * 8).reshape((3, 4, 5, 6, 7, 8)), mstype.float32),
                               name="x")
        self.value = value

    def construct(self, index_0, index_1, index_2):
        self.param[0:2, index_0, index_1, ..., index_2, 2] = self.value
        ret = self.param + self.const
        return ret


class TensorSetItemByMixedTensors_2(Cell):
    def __init__(self, value):
        super().__init__()
        self.const = Tensor(np.ones((3, 4, 5, 6, 7, 8), np.float16))
        self.param = Parameter(Tensor(np.arange(3 * 4 * 5 * 6 * 7 * 8).reshape((3, 4, 5, 6, 7, 8)), mstype.float16),
                               name="x")
        self.value = value

    def construct(self, index_0, index_1, index_2):
        self.param[..., index_0, index_1, index_2, 3] = self.value
        ret = self.param + self.const
        return ret


class TensorGetItemByMixedTensorsTypeError(Cell):
    def construct(self, x, index_0, index_1):
        ret = x[index_0, index_1, 0:3, ..., 0:5, [1, 2, 3, 4]]
        return ret


class TensorGetItemByMixedTensorsNumberError(Cell):
    def construct(self, x, index_0, index_1):
        ret = x[index_0, index_1, 0:3, ..., index_1, index_0]
        return ret


class TensorSetItemByOneTensorWithNumber(Cell):
    def __init__(self, value):
        super().__init__()
        self.const = Tensor(np.ones((6, 7, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.float32), name="x")
        self.value = value

    def construct(self, index):
        self.param[index] = self.value
        ret = self.param + self.const
        return ret


class TensorSetItemByOneTensorWithTensor(Cell):
    def __init__(self):
        super().__init__()
        self.const = Tensor(np.ones((6, 7, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.float32), name="x")

    def construct(self, index, value):
        self.param[index] = value
        ret = self.param + self.const
        return ret


class TensorSetItemByOneTensorWithTupleOfNumber(Cell):
    def __init__(self, value):
        super().__init__()
        self.const = Tensor(np.ones((6, 7, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.float32), name="x")
        self.value = value

    def construct(self, index):
        self.param[index] = self.value
        ret = self.param + self.const
        return ret


class TensorSetItemByOneTensorWithTupleOfTensor(Cell):
    def __init__(self):
        super().__init__()
        self.const = Tensor(np.ones((6, 3, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 3 * 8).reshape((6, 3, 8)), mstype.float32), name="x")

    def construct(self, index, value_0, value_1, value_2):
        self.param[index] = (value_0, value_1, value_2)
        ret = self.param + self.const
        return ret


class TensorSetItemByTensorsWithNumber(Cell):
    def __init__(self, value):
        super().__init__()
        self.const = Tensor(np.ones((6, 7, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.float32), name="x")
        self.value = value

    def construct(self, index_0, index_1, index_2):
        self.param[index_0, index_1, index_2] = self.value
        ret = self.param + self.const
        return ret


class TensorSetItemByTensorsWithTensor(Cell):
    def __init__(self):
        super().__init__()
        self.const = Tensor(np.ones((6, 7, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.float32), name="x")

    def construct(self, index_0, index_1, index_2, value):
        self.param[index_0, index_1, index_2] = value
        ret = self.param + self.const
        return ret


class TensorSetItemByTensorsWithTensorNumberError(Cell):
    def __init__(self):
        super().__init__()
        self.const = Tensor(np.ones((6, 7, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.float32), name="x")

    def construct(self, index_0, index_1, index_2, index_3, value):
        self.param[index_0, index_1, index_2, index_3] = value
        ret = self.param + self.const
        return ret


class TensorSetItemByTensorsWithTupleOfNumber(Cell):
    def __init__(self, value):
        super().__init__()
        self.const = Tensor(np.ones((6, 7, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.float32), name="x")
        self.value = value

    def construct(self, index_0, index_1, index_2):
        self.param[index_0, index_1, index_2] = self.value
        ret = self.param + self.const
        return ret


class TensorSetItemByTensorsWithTupleOfTensor(Cell):
    def __init__(self):
        super().__init__()
        self.const = Tensor(np.ones((6, 7, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.float32), name="x")

    def construct(self, index_0, index_1, index_2, value_0, value_1, value_2):
        self.param[index_0, index_1, index_2] = (value_0, value_1, value_2)
        ret = self.param + self.const
        return ret


class TensorSetItemByTensorsWithTupleOfTensorNumberError(Cell):
    def __init__(self):
        super().__init__()
        self.const = Tensor(np.ones((6, 7, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.float32), name="x")

    def construct(self, index_0, index_1, index_2, value_0, value_1):
        self.param[index_0, index_1, index_2] = (value_0, value_1)
        ret = self.param + self.const
        return ret


class TensorSetItemByMixedTensors(Cell):
    def __init__(self):
        super().__init__()
        self.const = Tensor(np.ones((6, 7, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.float32), name="x")
        self.value = 99.0

    def construct(self, index_0, index_1):
        self.param[index_0, index_1, 0:6] = self.value
        ret = self.param + self.const
        return ret


def test_tensor_assign():
    context.set_context(mode=context.GRAPH_MODE)
    net = TensorAssignWithSlice()
    net2 = TensorAssignWithSlice2()
    # The test case is no longer appropriate since x[1:3:-1] = np.array(2) does
    # not incur an error in numpy, which leaves the original array unchanged after
    # the assign operation.
    # net_e1 = TensorAssignWithSliceError1()
    # net_e2 = TensorAssignWithSliceError2()
    a = np.arange(60).reshape(3, 4, 5)
    ck = np.arange(60).reshape(3, 4, 5)
    b = Tensor([1], dtype=mstype.float32)
    Ta = Tensor(a, dtype=mstype.float32)
    Tck = Tensor(ck, dtype=mstype.float32)
    Ta4d = Tensor(a.reshape(1, 3, 4, 5), dtype=mstype.float32)
    Ta4d_ck = Tensor(ck.reshape(1, 3, 4, 5), dtype=mstype.float32)
    Tb = Tensor([1, 3], dtype=mstype.float32)
    t = Tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=mstype.float32)
    tck = Tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=mstype.float32)
    net(Ta, b, Tck)
    net2(t, b, tck)
    # Error for A[Slice] = Number
    # 1. A[Slice] = Number,  0 in shape

    # with pytest.raises(ValueError):
    #     net_e2(t, Tensor(2, mstype.int32))
    #      _pynative_executor.sync()

    # Error for A[Slice] = U, U is a Tensor
    # 1. A[Slice] = U,  u.size is error
    with pytest.raises(ValueError):
        net2(t, Tb, tck)
        _pynative_executor.sync()

    # 3. A[Slice] = U, U.size error
    with pytest.raises(ValueError):
        net2(t, Tb, tck)
        _pynative_executor.sync()

    # Error for A[Tuple(Slice...)] = Tensor
    # 2. A[Tuple(Slice...)] = U, U.size error
    with pytest.raises(ValueError):
        net(Ta, Tb, Tck)
    # 3. A[Tuple(Slice...)] = U,  Slice error
    # with pytest.raises(IndexError):
    #     net_e1(Ta, b)
    #      _pynative_executor.sync()

    # Error for A[Tuple(Slice...)] = Number
    # 1. A[Tuple(Slice...)] = Number,  Slice error
    # with pytest.raises(IndexError):
    #     net_e1(Ta, Tensor(2, mstype.int32))
    #      _pynative_executor.sync()

    net = TensorAssignWithInteger()
    # Error for A[Number] = scalar/Tensor
    # 1. A[Number] = U, U is a Tensor, u.size not match
    with pytest.raises(ValueError):
        net(Ta, Tb, Tck)
    # 2. A[Number] = U, the number index error
    with pytest.raises(IndexError):
        net(Ta4d, b, Ta4d_ck)
        _pynative_executor.sync()

    # Error for A[(n,m)] = scalar/Tensor
    # 1. A[(n,m)] = U, U is a tensor. u.size not match
    net = TensorAssignWithTupleInteger()
    with pytest.raises(ValueError):
        net(Ta, Tb, Tck)
    # 2. A[(n,m)] = U, the number index error
    with pytest.raises(IndexError):
        net(Ta4d, b, Ta4d_ck)
        _pynative_executor.sync()

    # Error for  A[...] = U or A[1:, ...] = u
    # 1. A[...] = scalar/tensor
    net = TensorAssignWithEllipsis()
    net(Ta, Ta4d)
    with pytest.raises(ValueError):
        net(Ta, Tb)
    # 2. A[::, 1:, ...] = scalar/tensor
    net = TensorAssignWithTupleEllipsis()
    net(Ta, b)
    Tc = Tensor(1, mstype.float32)
    net(Ta, Tc)
    with pytest.raises(ValueError):
        net(Ta, Tb)
        _pynative_executor.sync()


class TensorAssignWithTupleEllipsis2(Cell):
    def construct(self, a, b):
        a[1:, ..., ::] = b
        return a


class TensorAssignWithTupleEllipsis(Cell):
    def construct(self, a, b):
        a[:2, ...] = 1.0
        a[1:, ...] = b
        return a


class TensorAssignWithEllipsis(Cell):
    def construct(self, a, b):
        a[...] = 1
        a[...] = b
        return a


class TensorAssignWithInteger(Cell):
    def construct(self, a, b, ck):
        a[1] = 1
        a[0] = b
        z = a + ck
        return z


class TensorAssignWithTupleInteger(Cell):
    def construct(self, a, b, ck):
        a[(1)] = 1.0
        a[(1)] = b
        a[(1, 1)] = b
        a[(1, 1)] = 1.0
        z = a + ck
        return z


class TensorAssignWithBoolTensorIndex(Cell):
    def __init__(self):
        super().__init__()
        self.t = Tensor(np.arange(60).reshape([3, 4, 5]), dtype=mstype.float32)
        self.u_scalar = 5

    def construct(self, a, b, c, u_tensor):
        a[c] = self.u_scalar
        a[b] = u_tensor
        z = a + self.t
        return z


class TensorAssignWithBoolTensorIndexError(Cell):
    def construct(self, a, b, c, u_tensor):
        a[b][c] = u_tensor
        return a


class TensorAssignWithBoolTensorIndex2(Cell):
    def __init__(self):
        super().__init__()
        self.t = Tensor(np.arange(6).reshape([2, 3]), dtype=mstype.float32)
        self.t = Tensor(np.arange(60).reshape([3, 4, 5]), dtype=mstype.float32)
        self.u_scalar = 5

    def construct(self, a, u_tensor):
        a[a > 8] = u_tensor
        a[a >= 6] = self.u_scalar
        a[a < 3] = self.u_scalar
        a[a <= 5] = u_tensor
        a[a == 5] = self.u_scalar
        z = a + self.t
        return z


class TensorAssignWithBoolTensorIndex2Error(Cell):
    def construct(self, a, u_tensor):
        a[a > 8][a > 5] = u_tensor
        return a


class TensorItemSetWithNumber(Cell):
    def construct(self, tensor, number_value):
        ret = tensor.itemset(number_value)
        return ret


class TensorItemSetByItemWithNumber(Cell):
    def construct(self, tensor, index, number_value):
        ret = tensor.itemset(index, number_value)
        return ret


input_1d_np = np.ndarray([1]).astype(np.float32)
input_1d_ms = Tensor(input_1d_np, mstype.float32)

input_3d_np = np.random.randint(3, size=(3, 4, 5)).astype(np.int32)
input_3d_ms = Tensor(input_3d_np, mstype.float32)

index_np_1, index_np_2, index_np_3, index_np_4 = 0, 30, 60, 2.0
tuple_index_np_1, tuple_index_np_2, tuple_index_np_3, tuple_index_np_4, tuple_index_np_5 = \
    (0,), (1, 2), (1, 2, 3), (3, 4, 4), (1, 2, 3, 4)
value_np_1, value_np_2 = 1, 2.0

a = np.arange(60).reshape(3, 4, 5)
ck = np.arange(60).reshape(3, 4, 5)
a4 = np.arange(60).reshape(3, 2, 2, 5)
b = a > 5
c = a < 3
Ta = Tensor(a, dtype=mstype.float32)
Tck = Tensor(ck, dtype=mstype.float32)
Ta4 = Tensor(a4, dtype=mstype.float32)
Tb = Tensor(b)
Tc = Tensor(c)
Td = Tensor([True, True])
u_tensor = Tensor([1], dtype=mstype.float32)
u_tensor_error = Tensor([1, 2], dtype=mstype.float32)
t_1d = Tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=mstype.float32)
tck_1d = Tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=mstype.float32)
u_scalar = 5


def test_tensor_assign_bool_index():
    net1 = TensorAssignWithBoolTensorIndex()
    net2 = TensorAssignWithBoolTensorIndex2()
    net1(Ta, Tb, Tc, u_tensor)
    net1(Ta, Tb, Tc, u_tensor)
    with pytest.raises(ValueError):
        net1(Ta, Td, Tc, u_tensor)
    with pytest.raises(IndexError):
        net1(Ta, u_tensor, Tc, u_tensor)
    with pytest.raises(ValueError):
        net1(Ta, Tb, Td, u_tensor)
    with pytest.raises(IndexError):
        net1(Ta, Tb, Ta, u_tensor)
    with pytest.raises(ValueError):
        net1(Ta, Tb, Tc, u_tensor_error)
    # net1(Ta, u_tensor, Tc, u_tensor_error, u_scalar)
    with pytest.raises(ValueError):
        net2(Ta, u_tensor_error)
    net3 = TensorAssignWithBoolTensorIndexError()
    with pytest.raises(IndexError):
        net3(Ta, Tb, Tc, u_tensor)
    with pytest.raises(IndexError):
        net3(Ta, Tb, Tc, Tensor(u_scalar, mstype.int32))
    net4 = TensorAssignWithBoolTensorIndex2Error()
    with pytest.raises(IndexError):
        net4(Ta, u_tensor)
    with pytest.raises(IndexError):
        net4(Ta, Tensor(u_scalar, mstype.int32))
        _pynative_executor.sync()


test_cases = [
    ('TensorAssignWithTupleEllipsis2', {
        'block': TensorAssignWithTupleEllipsis2(),
        'desc_inputs': [Ta4, u_tensor],
    }),
    ('TensorAssignWithTupleEllipsis', {
        'block': TensorAssignWithTupleEllipsis(),
        'desc_inputs': [Ta, u_tensor],
    }),
    ('TensorAssignWithEllipsis', {
        'block': TensorAssignWithEllipsis(),
        'desc_inputs': [Ta, u_tensor],
    }),
    ('TensorAssignWithTupleInteger', {
        'block': TensorAssignWithTupleInteger(),
        'desc_inputs': [Ta, u_tensor, Tck],
    }),
    ('TensorAssignWithInteger', {
        'block': TensorAssignWithInteger(),
        'desc_inputs': [Ta, u_tensor, Tck],
    }),
    ('TensorAssignWithSlice', {
        'block': TensorAssignWithSlice(),
        'desc_inputs': [Ta, u_tensor, Tck],
    }),
    ('TensorAssignWithSlice2', {
        'block': TensorAssignWithSlice2(),
        'desc_inputs': [t_1d, u_tensor, tck_1d],
    }),
    ('TensorAssignWithBoolTensorIndex', {
        'block': TensorAssignWithBoolTensorIndex(),
        'desc_inputs': [Ta, Tb, Tc, u_tensor],
    }),
    ('TensorAssignWithBoolTensorIndex2', {
        'block': TensorAssignWithBoolTensorIndex2(),
        'desc_inputs': [Ta, u_tensor],
    }),
    ('SlicePositive', {
        'block': NetWorkSlicePositive(),
        'desc_inputs': [Tensor(np.ones([6, 8, 10], np.int32))],
    }),
    ('SliceReduceDimension', {
        'block': NetWorkReduceDimension(),
        'desc_inputs': [Tensor(np.ones([6, 8, 10], np.int32))],
    }),
    ('SliceNegative', {
        'block': NetWorkStepNegative(),
        'desc_inputs': [Tensor(np.ones([6, 8, 10], np.int32))],
    }),
    ('SliceReduceToScalar', {
        'block': NetWorkReduceToScalar(),
        'desc_inputs': [Tensor(np.ones([6, 8, 10], np.int32))],
    }),
    ('TensorSliceEllipsis', {
        'block': NetWorkSliceEllipsis(),
        'desc_inputs': [Tensor(np.ones([6, 7, 8, 9], np.int32))],
    }),
    ('TensorGetItemByOneTensor', {
        'block': TensorGetItemByOneTensor(),
        'desc_inputs': [Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.int32),
                        Tensor(np.random.randint(6, size=(5, 4)), mstype.int32)],
    }),
    ('TensorGetItemByTwoTensors', {
        'block': TensorGetItemByTwoTensors(),
        'desc_inputs': [Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.int32),
                        Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(7, size=(4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByThreeTensors', {
        'block': TensorGetItemByThreeTensors(),
        'desc_inputs': [Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.int32),
                        Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(7, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(8, size=(5, 3, 4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByMixedTensors_0', {
        'block': TensorGetItemByMixedTensors_0(),
        'desc_inputs': [Tensor(np.arange(3 * 4 * 5 * 6 * 7 * 8).reshape((3, 4, 5, 6, 7, 8)), mstype.float32),
                        Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByMixedTensors_1', {
        'block': TensorGetItemByMixedTensors_1(),
        'desc_inputs': [Tensor(np.arange(3 * 4 * 5 * 6 * 7 * 8).reshape((3, 4, 5, 6, 7, 8)), mstype.float32),
                        Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByMixedTensors_2', {
        'block': TensorGetItemByMixedTensors_2(),
        'desc_inputs': [Tensor(np.arange(3 * 4 * 5 * 6 * 7 * 8).reshape((3, 4, 5, 6, 7, 8)), mstype.float32),
                        Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByMixedTensors_3', {
        'block': TensorGetItemByMixedTensors_3(),
        'desc_inputs': [Tensor(np.arange(3 * 4 * 5 * 6 * 7 * 8).reshape((3, 4, 5, 6, 7, 8)), mstype.float32),
                        Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByMixedTensors_4', {
        'block': TensorGetItemByMixedTensors_4(),
        'desc_inputs': [Tensor(np.arange(3 * 4 * 5 * 6 * 7 * 8 * 9).reshape((3, 4, 5, 6, 7, 8, 9)), mstype.float32),
                        Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByMixedTensors_5', {
        'block': TensorGetItemByMixedTensors_5(),
        'desc_inputs': [Tensor(np.arange(3 * 4 * 5 * 6 * 7 * 8).reshape((3, 4, 5, 6, 7, 8)), mstype.float32),
                        Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByMixedTensors_6', {
        'block': TensorGetItemByMixedTensors_6(),
        'desc_inputs': [Tensor(np.arange(3 * 4 * 5 * 6 * 7 * 8).reshape((3, 4, 5, 6, 7, 8)), mstype.float32),
                        Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('TensorSetItemByOneTensorWithNumber', {
        'block': TensorSetItemByOneTensorWithNumber(value=0.0),
        'desc_inputs': [Tensor(np.random.randint(4, size=(5, 4)), mstype.int32)],
    }),
    ('TensorSetItemByOneTensorWithTensor', {
        'block': TensorSetItemByOneTensorWithTensor(),
        'desc_inputs': [Tensor(np.random.randint(3, size=(5, 4)), mstype.int32),
                        Tensor(np.zeros((4, 7, 8)), mstype.float32)],
    }),
    ('TensorSetItemByOneTensorWithTupleOfNumber', {
        'block': TensorSetItemByOneTensorWithTupleOfNumber(value=(0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7)),
        'desc_inputs': [Tensor(np.random.randint(5, size=(5, 4)), mstype.int32)],
    }),
    ('TensorSetItemByOneTensorWithTupleOfTensor', {
        'block': TensorSetItemByOneTensorWithTupleOfTensor(),
        'desc_inputs': [Tensor(np.random.randint(6, size=(5, 4)), mstype.int32),
                        Tensor(np.zeros((8,), np.float32)),
                        Tensor(np.ones((8,), np.float32)),
                        Tensor(np.ones((8,), np.float32) * 2)],
    }),
    ('TensorSetItemByTensorsWithNumber', {
        'block': TensorSetItemByTensorsWithNumber(value=0.0),
        'desc_inputs': [Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(7, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(8, size=(5, 3, 4, 5)), mstype.int32)],
    }),
    ('TensorSetItemByTensorsWithTensor', {
        'block': TensorSetItemByTensorsWithTensor(),
        'desc_inputs': [Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(7, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(8, size=(5, 3, 4, 5)), mstype.int32),
                        Tensor(np.zeros((4, 5)), mstype.float32)],
    }),
    ('TensorSetItemByTensorsWithTupleOfNumber', {
        'block': TensorSetItemByTensorsWithTupleOfNumber(value=(0.0, 1.1, 2.2, 3.3, 4.4)),
        'desc_inputs': [Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(7, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(8, size=(5, 3, 4, 5)), mstype.int32)],
    }),
    ('TensorSetItemByTensorsWithTupleOfTensor', {
        'block': TensorSetItemByTensorsWithTupleOfTensor(),
        'desc_inputs': [Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(7, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(8, size=(5, 3, 4, 5)), mstype.int32),
                        Tensor(np.zeros((4, 5)), mstype.float32),
                        Tensor(np.ones((4, 5)), mstype.float32),
                        Tensor(np.ones((4, 5)) * 2, mstype.float32)],
    }),
    ('TensorSetItemByMixedTensorsWithNumber_0', {
        'block': TensorSetItemByMixedTensors_0(value=88.0),
        'desc_inputs': [Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('TensorSetItemByMixedTensorsWithTensor_0', {
        'block': TensorSetItemByMixedTensors_0(value=Tensor(np.ones((4, 5, 3, 9), np.float32))),
        'desc_inputs': [Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByMixedTensorsWithTupleOfNumber_0', {
        'block': TensorSetItemByMixedTensors_0(value=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)),
        'desc_inputs': [Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByMixedTensorsWithTupleOfTensor_0', {
        'block': TensorSetItemByMixedTensors_0(value=(Tensor(np.ones((4, 5, 3, 9), np.float32)),
                                                      Tensor(np.zeros((4, 5, 3, 9), np.float32)),
                                                      Tensor(np.ones((4, 5, 3, 9), np.float32)))),
        'desc_inputs': [Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('TensorSetItemByMixedTensorsWithNumber_1', {
        'block': TensorSetItemByMixedTensors_1(value=88.0),
        'desc_inputs': [Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('TensorSetItemByMixedTensorsWithTensor_1', {
        'block': TensorSetItemByMixedTensors_1(value=Tensor(np.ones((5, 2, 6), np.float32))),
        'desc_inputs': [Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByMixedTensorsWithTupleOfNumber_1', {
        'block': TensorSetItemByMixedTensors_1(value=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
        'desc_inputs': [Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByMixedTensorsWithTupleOfTensor_1', {
        'block': TensorSetItemByMixedTensors_1(value=(Tensor(np.ones((5, 2, 6), np.float32)),
                                                      Tensor(np.zeros((5, 2, 6), np.float32)),
                                                      Tensor(np.ones((5, 2, 6), np.float32)),
                                                      Tensor(np.ones((5, 2, 6), np.float32)))),
        'desc_inputs': [Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('TensorSetItemByMixedTensorsWithNumber_2', {
        'block': TensorSetItemByMixedTensors_2(value=88.0),
        'desc_inputs': [Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('TensorSetItemByMixedTensorsWithTensor_2', {
        'block': TensorSetItemByMixedTensors_2(value=Tensor(np.ones((3, 4, 2, 3, 4, 5), np.float16))),
        'desc_inputs': [Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByMixedTensorsWithTupleOfNumber_2', {
        'block': TensorSetItemByMixedTensors_2(value=(1.0, 2.0, 3.0, 4.0, 5.0)),
        'desc_inputs': [Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByMixedTensorsWithTupleOfTensor_2', {
        'block': TensorSetItemByMixedTensors_2(value=(Tensor(np.ones((4, 5), np.float16)),
                                                      Tensor(np.zeros((4, 5), np.float16)),
                                                      Tensor(np.ones((4, 5), np.float16)))),
        'desc_inputs': [Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('1dTensorItemSetWithInt', {
        'block': TensorItemSetWithNumber(),
        'desc_inputs': [input_1d_ms, value_np_1]
    }),
    ('1dTensorItemSetWithFloat', {
        'block': TensorItemSetWithNumber(),
        'desc_inputs': [input_1d_ms, value_np_2]
    }),
    ('1dTensorItemSetByIntWithInt', {
        'block': TensorItemSetByItemWithNumber(),
        'desc_inputs': [input_1d_ms, index_np_1, value_np_1]
    }),
    ('1dTensorItemSetByIntWithFloat', {
        'block': TensorItemSetByItemWithNumber(),
        'desc_inputs': [input_1d_ms, index_np_1, value_np_2]
    }),
    ('3dTensorItemSetByIntWithInt', {
        'block': TensorItemSetByItemWithNumber(),
        'desc_inputs': [input_3d_ms, index_np_1, value_np_1]
    }),
    ('3dTensorItemSetByIntWithFloat', {
        'block': TensorItemSetByItemWithNumber(),
        'desc_inputs': [input_3d_ms, index_np_1, value_np_2]
    }),
    ('3dTensorItemSetByIntWithInt2', {
        'block': TensorItemSetByItemWithNumber(),
        'desc_inputs': [input_3d_ms, index_np_2, value_np_1]
    }),
    ('3dTensorItemSetByIntWithFloat2', {
        'block': TensorItemSetByItemWithNumber(),
        'desc_inputs': [input_3d_ms, index_np_2, value_np_2]
    }),
    ('1dTensorItemSetBy1dTupleWithInt', {
        'block': TensorItemSetByItemWithNumber(),
        'desc_inputs': [input_1d_ms, tuple_index_np_1, value_np_1]
    }),
    ('1dTensorItemSetBy1dTupleWithFloat', {
        'block': TensorItemSetByItemWithNumber(),
        'desc_inputs': [input_1d_ms, tuple_index_np_1, value_np_2]
    }),
    ('3dTensorItemSetBy3dTupleWithInt', {
        'block': TensorItemSetByItemWithNumber(),
        'desc_inputs': [input_3d_ms, tuple_index_np_3, value_np_1]
    }),
    ('3dTensorItemSetBy3dTupleWithFloat', {
        'block': TensorItemSetByItemWithNumber(),
        'desc_inputs': [input_3d_ms, tuple_index_np_3, value_np_2]
    }),
]

test_error_cases = [
    ('TensorGetItemByOneTensorDtypeError', {
        'block': (TensorGetItemByOneTensor(), {'exception': IndexError}),
        'desc_inputs': [Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.int32),
                        Tensor(np.random.randint(6, size=(5, 4)), mstype.int8)],
    }),
    ('TensorGetItemByTwoTensorsShapeError', {
        'block': (TensorGetItemByTwoTensors(), {'exception': IndexError}),
        'desc_inputs': [Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.int32),
                        Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(7, size=(2, 3, 5)), mstype.int32)],
    }),
    ('TensorGetItemByTwoTensorsDtypeError', {
        'block': (TensorGetItemByTwoTensors(), {'exception': IndexError}),
        'desc_inputs': [Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.int32),
                        Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(7, size=(4, 5)), mstype.float32)],
    }),
    ('TensorGetItemByThreeTensorsShapeError', {
        'block': (TensorGetItemByThreeTensors(), {'exception': IndexError}),
        'desc_inputs': [Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.int32),
                        Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(7, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(8, size=(5, 2, 4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByThreeTensorsDtypeError', {
        'block': (TensorGetItemByThreeTensors(), {'exception': IndexError}),
        'desc_inputs': [Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.int32),
                        Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(7, size=(4, 5)), mstype.int64),
                        Tensor(np.random.randint(8, size=(5, 3, 4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByMixedTensorsNumberError', {
        'block': (TensorGetItemByMixedTensorsNumberError(), {'exception': IndexError}),
        'desc_inputs': [Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.int32),
                        Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(7, size=(3, 4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByMixedTensorsTypeError', {
        'block': (TensorGetItemByMixedTensorsTypeError(), {'exception': IndexError}),
        'desc_inputs': [Tensor(np.arange(3 * 4 * 5 * 6 * 7 * 8 * 9).reshape((3, 4, 5, 6, 7, 8, 9)), mstype.int32),
                        Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(3, 4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByMixedTensorsDtypeError', {
        'block': (TensorGetItemByMixedTensors_0(), {'exception': IndexError}),
        'desc_inputs': [Tensor(np.arange(3 * 4 * 5 * 6 * 7 * 8 * 9).reshape((3, 4, 5, 6, 7, 8, 9)), mstype.int32),
                        Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.float32)],
    }),
    ('TensorGetItemByMixedTensorsShapeError', {
        'block': (TensorGetItemByMixedTensors_0(), {'exception': IndexError}),
        'desc_inputs': [Tensor(np.arange(3 * 4 * 5 * 6 * 7 * 8 * 9).reshape((3, 4, 5, 6, 7, 8, 9)), mstype.int32),
                        Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(2, 4, 5)), mstype.int32)],
    }),
    ('TensorSetItemByOneTensorWithNumberTypeError', {
        'block': (TensorSetItemByOneTensorWithNumber(value=0), {'exception': TypeError}),
        'desc_inputs': [Tensor(np.random.randint(4, size=(5, 4)), mstype.int32)],
    }),
    ('TensorSetItemByOneTensorWithTensorShapeError', {
        'block': (TensorSetItemByOneTensorWithTensor(), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.random.randint(3, size=(5, 4)), mstype.int32),
                        Tensor(np.zeros((6, 7, 8)), mstype.float32)],
    }),
    ('TensorSetItemByOneTensorWithTensorDtypeError', {
        'block': (TensorSetItemByOneTensorWithTensor(), {'exception': TypeError}),
        'desc_inputs': [Tensor(np.random.randint(3, size=(5, 4)), mstype.int32),
                        Tensor(np.zeros((6, 7, 8)), mstype.int32)],
    }),
    ('TensorSetItemByOneTensorWithTupleOfNumberTypeError', {
        'block': (TensorSetItemByOneTensorWithTupleOfNumber(value=(0, 1, 2, 3, 4, 5, 6, 7)), {'exception': TypeError}),
        'desc_inputs': [Tensor(np.random.randint(5, size=(5, 4)), mstype.int32)],
    }),
    ('TensorSetItemByOneTensorWithTupleOfNumberNumberError', {
        'block': (TensorSetItemByOneTensorWithTupleOfNumber(value=(0.0, 1.1, 2.2)), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.random.randint(5, size=(5, 4)), mstype.int32)],
    }),
    ('TensorSetItemByOneTensorWithTupleOfTensorDtyeError', {
        'block': (TensorSetItemByOneTensorWithTupleOfTensor(), {'exception': TypeError}),
        'desc_inputs': [Tensor(np.random.randint(6, size=(5, 4)), mstype.int32),
                        Tensor(np.zeros((8,), np.int32)),
                        Tensor(np.ones((8,), np.int32)),
                        Tensor(np.ones((8,), np.float32) * 2)],
    }),
    ('TensorSetItemByTensorsWithNumberTypeError', {
        'block': (TensorSetItemByTensorsWithNumber(value=0), {'exception': TypeError}),
        'desc_inputs': [Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(7, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(8, size=(5, 3, 4, 5)), mstype.int32)],
    }),
    ('TensorSetItemByTensorsWithTensorShapeError', {
        'block': (TensorSetItemByTensorsWithTensor(), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(7, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(8, size=(5, 3, 4, 5)), mstype.int32),
                        Tensor(np.zeros((2, 5)), mstype.float32)],
    }),
    ('TensorSetItemByTensorsWithTensorTypeError', {
        'block': (TensorSetItemByTensorsWithTensor(), {'exception': TypeError}),
        'desc_inputs': [Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(7, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(8, size=(5, 3, 4, 5)), mstype.int32),
                        Tensor(np.zeros((4, 5)), mstype.int32)],
    }),
    ('TensorSetItemByTensorsWithTensorNumberError', {
        'block': (TensorSetItemByTensorsWithTensorNumberError(), {'exception': IndexError}),
        'desc_inputs': [Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(7, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(8, size=(5, 3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(8, size=(1, 3, 4, 5)), mstype.int32),
                        Tensor(np.zeros((2, 5)), mstype.float32)],
    }),
    ('TensorSetItemByTensorsWithTupleOfNumberTypeError', {
        'block': (TensorSetItemByTensorsWithTupleOfNumber(value=(0.0, 1, 2, 3, 4)), {'exception': TypeError}),
        'desc_inputs': [Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(7, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(8, size=(5, 3, 4, 5)), mstype.int32)],
    }),
    ('TensorSetItemByTensorsWithTupleOfNumberNumberError', {
        'block': (TensorSetItemByTensorsWithTupleOfNumber(value=(0.0, 1.0, 2.0, 3.0)), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(7, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(8, size=(5, 3, 4, 5)), mstype.int32)],
    }),
    ('TensorSetItemByTensorsWithTupleOfTensorNumberError', {
        'block': (TensorSetItemByTensorsWithTupleOfTensorNumberError(), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(7, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(8, size=(5, 3, 4, 5)), mstype.int32),
                        Tensor(np.zeros((4, 5)), mstype.float32),
                        Tensor(np.ones((4, 5)), mstype.float32)],
    }),
    ('TensorSetItemByTensorsWithTupleOfTensorTypeError', {
        'block': (TensorSetItemByTensorsWithTupleOfTensor(), {'exception': TypeError}),
        'desc_inputs': [Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(7, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(8, size=(5, 3, 4, 5)), mstype.int32),
                        Tensor(np.zeros((4, 5)), mstype.float32),
                        Tensor(np.ones((4, 5)), mstype.int32),
                        Tensor(np.ones((4, 5)) * 2, mstype.int32)],
    }),
    ('TensorSetItemByMixedTensorsWithNumberValueTypeError', {
        'block': (TensorSetItemByMixedTensors_1(value=88), {'exception': TypeError}),
        'desc_inputs': [Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('TensorSetItemByMixedTensorsWithNumberIndexTypeError', {
        'block': (TensorSetItemByMixedTensors_1(value=88.0), {'exception': IndexError}),
        'desc_inputs': [Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.float32)],
    }),
    ('TensorSetItemByMixedTensorsWithTensorValueDtypeError', {
        'block': (TensorSetItemByMixedTensors_1(value=Tensor(np.ones((5, 2, 6), np.int32))),
                  {'exception': TypeError}),
        'desc_inputs': [Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('TensorSetItemByMixedTensorsWithTensorValueShapeError', {
        'block': (TensorSetItemByMixedTensors_1(value=Tensor(np.ones((3, 2, 6), np.float32))),
                  {'exception': ValueError}),
        'desc_inputs': [Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('TensorSetItemByMixedTensorsWithTensorIndexDtypeError', {
        'block': (TensorSetItemByMixedTensors_1(value=Tensor(np.ones((5, 2, 6), np.float32))),
                  {'exception': IndexError}),
        'desc_inputs': [Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.float32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByMixedTensorsWithTupleOfNumberValueTypeError', {
        'block': (TensorSetItemByMixedTensors_1(value=(1.0, 2, 3.0, 4.0, 5.0, 6.0)),
                  {'exception': TypeError}),
        'desc_inputs': [Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByMixedTensorsWithTupleOfTensorValueDtypeError', {
        'block': (TensorSetItemByMixedTensors_1(value=(Tensor(np.ones((5, 2, 6), np.float32)),
                                                       Tensor(np.zeros((5, 2, 6), np.float32)),
                                                       Tensor(np.ones((5, 2, 6), np.float32)),
                                                       Tensor(np.ones((5, 2, 6), np.int32)))),
                  {'exception': TypeError}),
        'desc_inputs': [Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('TensorGetItemByMixedTensorsWithTupleOfTensorIndexDtypeError', {
        'block': (TensorSetItemByMixedTensors_1(value=(Tensor(np.ones((5, 2, 6), np.float32)),
                                                       Tensor(np.zeros((5, 2, 6), np.float32)),
                                                       Tensor(np.ones((5, 2, 6), np.float32)),
                                                       Tensor(np.ones((5, 2, 6), np.int32)))),
                  {'exception': IndexError}),
        'desc_inputs': [Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.float32),
                        Tensor(np.random.randint(4, size=(4, 5)), mstype.int32),
                        Tensor(np.random.randint(3, size=(2, 1, 4, 5)), mstype.int32)],
    }),
    ('3dTensorItemSetWithInt', {
        'block': (TensorItemSetWithNumber(), {'exception': IndexError}),
        'desc_inputs': [input_3d_ms, value_np_1]
    }),
    ('3dTensorItemSetWithFloat', {
        'block': (TensorItemSetWithNumber(), {'exception': IndexError}),
        'desc_inputs': [input_3d_ms, value_np_2]
    }),
    ('1dTensorItemSetByOverflowIntWithInt', {
        'block': (TensorItemSetByItemWithNumber(), {'exception': IndexError}),
        'desc_inputs': [input_1d_ms, index_np_2, value_np_1]
    }),
    ('1dTensorItemSetByOverflowIntWithFloat', {
        'block': (TensorItemSetByItemWithNumber(), {'exception': IndexError}),
        'desc_inputs': [input_1d_ms, index_np_2, value_np_2]
    }),
    ('1dTensorItemSetByFloatWithInt', {
        'block': (TensorItemSetByItemWithNumber(), {'exception': TypeError}),
        'desc_inputs': [input_1d_ms, index_np_4, value_np_1]
    }),
    ('1dTensorItemSetByFLoatWithFloat', {
        'block': (TensorItemSetByItemWithNumber(), {'exception': TypeError}),
        'desc_inputs': [input_1d_ms, index_np_4, value_np_2]
    }),
    ('3dTensorItemSetByOverflowIntWithInt', {
        'block': (TensorItemSetByItemWithNumber(), {'exception': IndexError}),
        'desc_inputs': [input_3d_ms, index_np_3, value_np_1]
    }),
    ('3dTensorItemSetByOverflowIntWithFloat', {
        'block': (TensorItemSetByItemWithNumber(), {'exception': IndexError}),
        'desc_inputs': [input_3d_ms, index_np_3, value_np_2]
    }),
    ('3dTensorItemSetByFloatIntWithInt', {
        'block': (TensorItemSetByItemWithNumber(), {'exception': TypeError}),
        'desc_inputs': [input_3d_ms, index_np_4, value_np_1]
    }),
    ('3dTensorItemSetByFloatWithFloat', {
        'block': (TensorItemSetByItemWithNumber(), {'exception': TypeError}),
        'desc_inputs': [input_3d_ms, index_np_4, value_np_2]
    }),
    ('1dTensorItemSetBy2dTupleWithFloat', {
        'block': (TensorItemSetByItemWithNumber(), {'exception': ValueError}),
        'desc_inputs': [input_1d_ms, tuple_index_np_2, value_np_1]
    }),
    ('1dTensorItemSetBy2dTupleWithFloat', {
        'block': (TensorItemSetByItemWithNumber(), {'exception': ValueError}),
        'desc_inputs': [input_1d_ms, tuple_index_np_2, value_np_2]
    }),
    ('3dTensorItemSetBy1dTupleWithFloat', {
        'block': (TensorItemSetByItemWithNumber(), {'exception': ValueError}),
        'desc_inputs': [input_3d_ms, tuple_index_np_1, value_np_1]
    }),
    ('3dTensorItemSetBy1dTupleWithFloat', {
        'block': (TensorItemSetByItemWithNumber(), {'exception': ValueError}),
        'desc_inputs': [input_3d_ms, tuple_index_np_1, value_np_2]
    }),
    ('3dTensorItemSetBy2dTupleWithFloat', {
        'block': (TensorItemSetByItemWithNumber(), {'exception': ValueError}),
        'desc_inputs': [input_3d_ms, tuple_index_np_2, value_np_1]
    }),
    ('3dTensorItemSetBy2dTupleWithFloat', {
        'block': (TensorItemSetByItemWithNumber(), {'exception': ValueError}),
        'desc_inputs': [input_3d_ms, tuple_index_np_2, value_np_2]
    }),
    ('3dTensorItemSetBy3dTupleOverFloawWithFloat', {
        'block': (TensorItemSetByItemWithNumber(), {'exception': ValueError}),
        'desc_inputs': [input_3d_ms, tuple_index_np_4, value_np_1]
    }),
    ('3dTensorItemSetBy3dTupleOverFloawWithFloat', {
        'block': (TensorItemSetByItemWithNumber(), {'exception': ValueError}),
        'desc_inputs': [input_3d_ms, tuple_index_np_4, value_np_2]
    }),
    ('3dTensorItemSetBy4dTupleWithFloat', {
        'block': (TensorItemSetByItemWithNumber(), {'exception': ValueError}),
        'desc_inputs': [input_3d_ms, tuple_index_np_5, value_np_1]
    }),
    ('3dTensorItemSetBy4dTupleWithFloat', {
        'block': (TensorItemSetByItemWithNumber(), {'exception': ValueError}),
        'desc_inputs': [input_3d_ms, tuple_index_np_5, value_np_2]
    })
]


@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config)
def test_exec():
    context.set_context(mode=context.GRAPH_MODE)
    return test_cases


@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config_exception)
def test_check_exception():
    return test_error_cases


def test_tensor_slice_reduce_out_of_bounds_neg():
    class NetWork(Cell):
        def __init__(self):
            super().__init__()
            self.tensor_ret = Tensor(np.array(9, np.int32))

        def construct(self, tensor):
            ret = tensor[-7, 3, 4]
            return ret

    input_tensor = Tensor(np.ones([6, 8, 10], np.int32))
    net = NetWork()
    with pytest.raises(IndexError):
        net(input_tensor)
        _pynative_executor.sync()


def test_tensor_slice_reduce_out_of_bounds_positive():
    class NetWork(Cell):
        def __init__(self):
            super().__init__()
            self.tensor_ret = Tensor(np.array(9, np.int32))

        def construct(self, tensor):
            ret = tensor[6, 3, 4]
            return ret

    input_tensor = Tensor(np.ones([6, 8, 10], np.int32))
    net = NetWork()
    with pytest.raises(IndexError):
        net(input_tensor)
        _pynative_executor.sync()


def test_tensor_slice_none_in_pynative():
    """
    Feature: Test Tensor slice None
    Description: test tensor slice success
    Expectation: success
    """
    x_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    x = Tensor(x_np)
    np.allclose(x[..., None].asnumpy(), x_np)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_gpu', 'cpu_linux'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
def test_tensor_index_augassign_01():
    """
    Feature: Test tensor index augassign
    Description: test tensor index augassign success
    Expectation: success
    """

    @jit(backend="ms_backend")
    def foo1(x):
        x[1, :] += 2
        return x

    def foo2(x):
        x[1, :] += 2
        return x

    input_np_x = np.ones([2, 2]).astype(np.float16)

    input_x1 = Tensor(input_np_x, mstype.float16)
    out1 = foo1(input_x1).asnumpy()

    input_x2 = Tensor(input_np_x, mstype.float16)
    out2 = foo2(input_x2).asnumpy()

    input_np_x[1, :] += 2
    out_numpy = input_np_x
    assert np.allclose(out1, out_numpy, 0.001, 0.001)
    assert np.allclose(out2, out_numpy, 0.001, 0.001)


class GradOfAllInputs(Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad_op = ops.GradOperation(get_all=True, sens_param=True)

    def construct(self, *inputs):
        grad_net = self.grad_op(self.net)
        return grad_net(*inputs)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_gpu', 'cpu_linux'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_tensor_index_augassign_02(mode):
    """
    Feature: Test tensor index augassign
    Description: test tensor index augassign success
    Expectation: success
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.input_x = Parameter(Tensor([2.0, 3.0, 4.0, 5.0], mstype.float32))
            self.relu = nn.ReLU()

        def construct(self, input_y):
            input_x = self.input_x * 1
            input_x[...] += input_y[0, :]
            out = self.relu(input_x)
            return out

    class NetPytorch(nn_torch.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn_torch.ReLU()

        def forward(self, input_y):
            input_x = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
            input_x[...] += input_y[0, :]
            out = self.relu(input_x)
            return out

    context.set_context(mode=mode, jit_level='O0')

    input_me_1 = Tensor([[3.4], [3.4]], mstype.float32)
    input_me_2 = Tensor([[1.1, 2.2, 3.3, 4.4]], mstype.float32)
    input_torch_1 = torch.tensor([[3.4], [3.4]], dtype=torch.float32)
    input_torch_2 = torch.tensor([[1.1, 2.2, 3.3, 4.4]], dtype=torch.float32)

    net = Net()
    net.set_grad()
    grad_net = GradOfAllInputs(net)
    grad_net.set_train()
    out_me_1 = net(input_me_1)
    grad_net(input_me_1, out_me_1)
    out_me_2 = net(input_me_2)

    net_torch = NetPytorch()
    out_torch_1 = net_torch(input_torch_1)
    out_torch_2 = net_torch(input_torch_2)
    assert np.allclose(out_torch_1, out_me_1.asnumpy(), 0.001, 0.001)
    assert np.allclose(out_torch_2, out_me_2.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize("value", [[2, 3, 4, 5, 6, 7, 8], [2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8],
                                   [2, 3.3, 4.4, 5.5, 6.6, True, False]])
def test_tensor_index_augassign_03(value):
    """
    Feature: Test tensor index augassign
    Description: test tensor index augassign success
    Expectation: success
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def construct(self, input_x):
            input_x = input_x * 1
            input_x[:, 2, 2, [1, 2, 3, 4], 2, None, ..., True] = (
                    value * input_x[:, 2, 2, [1, 2, 3, 4], 2, None, ..., True])
            out = self.relu(input_x)
            return out

    class NetPytorch(nn_torch.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn_torch.ReLU()

        def forward(self, input_x):
            input_x[:, 2, 2, [1, 2, 3, 4], 2, None, ..., True] *= value
            out = self.relu(torch.from_numpy(input_x))
            return out

    input_me_1 = Tensor(np.arange(2 * 3 * 4 * 5 * 6 * 7).reshape(2, 3, 4, 5, 6, 7), mstype.float16)
    input_me_2 = Tensor(np.arange(1 * 3 * 3 * 5 * 3 * 7).reshape(1, 3, 3, 5, 3, 7), mstype.float16)
    input_torch_1 = np.arange(2 * 3 * 4 * 5 * 6 * 7).reshape(2, 3, 4, 5, 6, 7).astype(np.float32)
    input_torch_2 = np.arange(1 * 3 * 3 * 5 * 3 * 7).reshape(1, 3, 3, 5, 3, 7).astype(np.float32)

    net = Net()
    net.construct = jit(net.construct, backend='ms_backend')
    net.set_grad()
    grad_net = GradOfAllInputs(net)
    grad_net.set_train()
    out_me_1 = net(input_me_1)
    grad_net(input_me_1, out_me_1)
    out_me_2 = net(input_me_2)

    net_torch = NetPytorch()
    out_torch_1 = net_torch(input_torch_1)
    out_torch_2 = net_torch(input_torch_2)
    assert np.allclose(out_torch_1, out_me_1.asnumpy(), 0.001, 0.001)
    assert np.allclose(out_torch_2, out_me_2.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend910b', 'cpu_linux'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_tensor_index_augassign_04(mode):
    """
    Feature: Test tensor index augassign
    Description: test tensor index augassign success
    Expectation: success
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.scalar = 5

        def construct(self, x):
            relu = self.relu(x)
            if self.scalar > 3:
                for _ in range(3):
                    x[2, :] += relu[3, :]
            else:
                for _ in range(2):
                    x[2, :] += relu[3, :]
            return x

    class NetPytorch(nn_torch.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn_torch.ReLU()

        def forward(self, input_x):
            x = torch.from_numpy(input_x)
            relu = self.relu(x)
            for _ in range(3):
                x[2, :] += relu[3, :]
            return x

    context.set_context(mode=mode, jit_level='O0')

    input_np_x = np.random.rand(4, 4)
    input_me_x = Tensor(input_np_x, mstype.float32)
    net = Net()
    out_me = net(input_me_x)

    net_torch = NetPytorch()
    out_torch = net_torch(input_np_x)

    assert np.allclose(out_torch, out_me.asnumpy(), 0.01, 0.01)


@arg_mark(plat_marks=['platform_ascend910b', 'cpu_linux'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_tensor_index_augassign_05(mode):
    """
    Feature: Test tensor index augassign
    Description: test tensor index augassign success
    Expectation: success
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(axis=-1)
            self.sigmoid = nn.Sigmoid()

        def construct(self, x):
            relu = self.relu(x)
            softmax = self.softmax(x)
            softmax[:, :, 3, :] /= relu[:, 2, :, :]
            return self.sigmoid(softmax)

    class NetPytorch(nn_torch.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn_torch.ReLU()
            self.softmax = nn_torch.Softmax(dim=-1)
            self.sigmoid = nn_torch.Sigmoid()

        def forward(self, input_x):
            x = torch.from_numpy(input_x)
            relu = self.relu(x)
            softmax = self.softmax(x)
            softmax[:, :, 3, :] /= relu[:, 2, :, :]
            return self.sigmoid(softmax)

    context.set_context(mode=mode, jit_level='O0')

    input_np_x = np.random.rand(4, 4, 4, 4)
    input_me_x = Tensor(input_np_x, mstype.float32)
    net = Net()
    out_me = net(input_me_x)

    net_torch = NetPytorch()
    out_torch = net_torch(input_np_x)

    assert np.allclose(out_torch, out_me.asnumpy(), 0.01, 0.01)
