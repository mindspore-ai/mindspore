# Copyright 2021-2023 Huawei Technologies Co., Ltd
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

"""Operators for reinforce learning."""

from functools import reduce
import mindspore.context as context
from mindspore import _checkparam as validator
from ...common import dtype as mstype
from ..primitive import prim_attr_register, PrimitiveWithInfer


class BufferSample(PrimitiveWithInfer):
    r"""
    In reinforcement learning, the data is sampled from the replaybuffer randomly.

    Returns the tuple tensor with the given shape, decided by the given batchsize.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        capacity (int64): Capacity of the buffer, must be non-negative.
        batch_size (int64): The size of the sampled data, lessequal to `capacity`.
        buffer_shape (tuple(shape)): The shape of an buffer.
        buffer_dtype (tuple(type)): The type of an buffer.
        seed (int64): Random seed for sample. Default: ``0`` . If use the default seed, it will generate a ramdom
        one in kernel. Set a number other than `0` to keep a specific seed. Default: ``0`` .
        unique (bool): Whether the sampled data is strictly unique. Setting it to False has a better performance.
            Default: ``False`` .

    Inputs:
        - **data** (tuple(Parameter(Tensor))) - The tuple(Tensor) represents replaybuffer,
          each tensor is described by the `buffer_shape` and `buffer_type`.
        - **count** (Parameter) - The count means the real available size of the buffer,
          data type: int32.
        - **head** (Parameter) - The position of the first data in buffer, data type: int32.

    Outputs:
        tuple(Tensor). The shape is `batch_size` * `buffer_shape`. The dtype is `buffer_dtype`.

    Raises:
        TypeError: If `buffer_shape` is not a tuple.
        ValueError: If batch_size is larger than capacity.
        ValueError: If `capacity` is not a positive integer.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> capacity = 100
        >>> batch_size = 5
        >>> count = Parameter(Tensor(5, ms.int32), name="count")
        >>> head = Parameter(Tensor(0, ms.int32), name="head")
        >>> shapes = [(4,), (2,), (1,), (4,)]
        >>> types = [ms.float32, ms.int32, ms.int32, ms.float32]
        >>> buffer = [Parameter(Tensor(np.arange(100 * 4).reshape(100, 4).astype(np.float32)), name="states"),
        ...           Parameter(Tensor(np.arange(100 * 2).reshape(100, 2).astype(np.int32)), name="action"),
        ...           Parameter(Tensor(np.ones((100, 1)).astype(np.int32)), name="reward"),
        ...           Parameter(Tensor(np.arange(100 * 4).reshape(100, 4).astype(np.float32)), name="state_")]
        >>> buffer_sample = ops.BufferSample(capacity, batch_size, shapes, types)
        >>> output = buffer_sample(buffer, count, head)
        >>> print(output)
            (Tensor(shape=[5, 4], dtype=Float32, value=
                [[ 0.00000000e+00, 1.00000000e+00, 2.00000000e+00, 3.00000000e+00],
                [ 8.00000000e+00, 9.00000000e+00, 1.00000000e+01, 1.10000000e+01],
                [ 1.60000000e+01, 1.70000000e+01, 1.80000000e+01, 1.90000000e+01],
                [ 1.20000000e+01, 1.30000000e+01, 1.40000000e+01, 1.50000000e+01],
                [ 3.20000000e+01, 3.30000000e+01, 3.40000000e+01, 3.50000000e+01]]),
             Tensor(shape=[5, 2], dtype=Int32, value=
                [[ 0, 1],
                [ 4, 5],
                [ 8, 9],
                [ 6, 7],
                [16, 17]]),
             Tensor(shape=[5, 1], dtype=Int32, value=
                [[1],
                [1],
                [1],
                [1],
                [1]]),
             Tensor(shape=[5, 4], dtype=Float32, value=
                [[ 0.00000000e+00, 1.00000000e+00, 2.00000000e+00, 3.00000000e+00],
                [ 8.00000000e+00, 9.00000000e+00, 1.00000000e+01, 1.10000000e+01],
                [ 1.60000000e+01, 1.70000000e+01, 1.80000000e+01, 1.90000000e+01],
                [ 1.20000000e+01, 1.30000000e+01, 1.40000000e+01, 1.50000000e+01],
                [ 3.20000000e+01, 3.30000000e+01, 3.40000000e+01, 3.50000000e+01]]))
    """

    @prim_attr_register
    def __init__(self, capacity, batch_size, buffer_shape, buffer_dtype, seed=0, unique=False):
        """Initialize BufferSample."""
        self.init_prim_io_names(inputs=["buffer"], outputs=["sample"])
        validator.check_value_type("shape of init data", buffer_shape, [tuple, list], self.name)
        validator.check_int(capacity, 1, validator.GE, "capacity", self.name)
        self._batch_size = batch_size
        self._buffer_shape = buffer_shape
        self._buffer_dtype = buffer_dtype
        self._n = len(buffer_shape)
        validator.check_int(self._batch_size, capacity, validator.LE, "batchsize", self.name)
        self.add_prim_attr('capacity', capacity)
        self.add_prim_attr('seed', seed)
        self.add_prim_attr('unique', unique)
        buffer_elements = []
        for shape in buffer_shape:
            buffer_elements.append(reduce(lambda x, y: x * y, shape))
        self.add_prim_attr('buffer_elements', buffer_elements)
        self.add_prim_attr('buffer_dtype', buffer_dtype)
        self.add_prim_attr('side_effect_mem', True)
        if context.get_context('device_target') == "Ascend":
            self.add_prim_attr('device_target', "CPU")

    def infer_shape(self, data_shape, count_shape, head_shape):
        validator.check_value_type("shape of data", data_shape, [tuple, list], self.name)
        out_shapes = []
        for i in range(self._n):
            out_shapes.append((self._batch_size,) + self._buffer_shape[i])
        return tuple(out_shapes)

    def infer_dtype(self, data_type, count_type, head_type):
        validator.check_type_name("count type", count_type, (mstype.int32), self.name)
        validator.check_type_name("head type", head_type, (mstype.int32), self.name)
        return tuple(self._buffer_dtype)


class BufferAppend(PrimitiveWithInfer):
    r"""
    In reinforcement learning, the experience data is collected in each step. We use `BufferAppend` to
    push data to the bottom of buffer under the First-In-First-Out rule.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        capacity (int64): Capacity of the buffer, must be non-negative.
        buffer_shape (tuple(shape)): The shape of an buffer.
        buffer_dtype (tuple(type)): The type of an buffer.

    Inputs:
        - **data** (tuple(Parameter(Tensor))) - The tuple(Tensor) represents replaybuffer,
          each tensor is described by the `buffer_shape` and `buffer_type`.
        - **exp** (tuple(Parameter(Tensor))) - The tuple(Tensor) represents one list of experience data,
          each tensor is described by the `buffer_shape` and `buffer_type`.
        - **count** (Parameter) - The count means the real available size of the buffer,
          data type: int32.
        - **head** (Parameter) - The position of the first data in buffer, data type: int32.

    Outputs:
        None.

    Raises:
        ValueError: If `count` and `head` is not an integer.
        ValueError: If `capacity` is not a positive integer.
        ValueError: If length of `data` is not equal to length of `exp`.
        ValueError: If dim of data is equal to dim of exp, but `data[1:]` is not equal to the shape in `exp`.
        ValueError: If the shape of `data[1:]` is not equal to the shape in `exp`.
        TypeError: If the type in `exp` is not the same with `data`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> capacity = 100
        >>> count = Parameter(Tensor(5, ms.int32), name="count")
        >>> head = Parameter(Tensor(0, ms.int32), name="head")
        >>> shapes = [(4,), (2,), (1,), (4,)]
        >>> types = [ms.float32, ms.int32, ms.int32, ms.float32]
        >>> buffer = [Parameter(Tensor(np.arange(100 * 4).reshape(100, 4).astype(np.float32)), name="states"),
        ...           Parameter(Tensor(np.arange(100 * 2).reshape(100, 2).astype(np.int32)), name="action"),
        ...           Parameter(Tensor(np.ones((100, 1)).astype(np.int32)), name="reward"),
        ...           Parameter(Tensor(np.arange(100 * 4).reshape(100, 4).astype(np.float32)), name="state_")]
        >>> exp = [Tensor(np.array([2, 2, 2, 2]), ms.float32), Tensor(np.array([0, 0]), ms.int32),
        ...        Tensor(np.array([0]), ms.int32), Tensor(np.array([3, 3, 3, 3]), ms.float32)]
        >>> batch_exp = [Tensor(np.array([[2, 2, 2, 2], [2, 2, 2, 2]]), ms.float32),
        ...              Tensor(np.array([[0, 0], [0, 0]]), ms.int32),
        ...              Tensor(np.array([[0], [0]]), ms.int32),
        ...              Tensor(np.array([[3, 3, 3, 3], [3, 3, 3, 3]]), ms.float32)]
        >>> buffer_append = ops.BufferAppend(capacity, shapes, types)
        >>> buffer_append(buffer, exp, count, head)
        >>> buffer_append(buffer, batch_exp, count, head)
    """
    @prim_attr_register
    def __init__(self, capacity, buffer_shape, buffer_dtype):
        """Initialize BufferAppend."""
        validator.check_int(capacity, 1, validator.GE, "capacity", self.name)
        self.add_prim_attr('capacity', capacity)
        buffer_elements = []
        for shape in buffer_shape:
            buffer_elements.append(reduce(lambda x, y: x * y, shape))
        self.add_prim_attr('buffer_elements', buffer_elements)
        self.add_prim_attr('buffer_dtype', buffer_dtype)
        self.add_prim_attr('side_effect_mem', True)
        if context.get_context('device_target') == "Ascend":
            self.add_prim_attr('device_target', "CPU")


class BufferGetItem(PrimitiveWithInfer):
    r"""
    Get the data from buffer in the position of input index.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        capacity (int64): Capacity of the buffer, must be non-negative.
        buffer_shape (tuple(shape)): The shape of an buffer.
        buffer_dtype (tuple(type)): The type of an buffer.

    Inputs:
        - **data** (tuple(Parameter(Tensor))) - The tuple(Tensor) represents replaybuffer,
          each tensor is described by the `buffer_shape` and `buffer_type`.
        - **count** (Parameter) - The count means the real available size of the buffer,
          data type: int32.
        - **head** (Parameter) - The position of the first data in buffer, data type: int32.
        - **index** (int64) - The position of the data in buffer.

    Outputs:
        tuple(Tensor). The shape is `buffer_shape`. The dtype is `buffer_dtype`.

    Raises:
        ValueError: If `count` and `head` is not an integer.
        ValueError: If `capacity` is not a positive integer.
        TypeError: If `buffer_shape` is not a tuple.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> capacity = 100
        >>> index = 3
        >>> count = Parameter(Tensor(5, ms.int32), name="count")
        >>> head = Parameter(Tensor(0, ms.int32), name="head")
        >>> shapes = [(4,), (2,), (1,), (4,)]
        >>> types = [ms.float32, ms.int32, ms.int32, ms.float32]
        >>> buffer = [Parameter(Tensor(np.arange(100 * 4).reshape(100, 4).astype(np.float32)), name="states"),
        ...           Parameter(Tensor(np.arange(100 * 2).reshape(100, 2).astype(np.int32)), name="action"),
        ...           Parameter(Tensor(np.ones((100, 1)).astype(np.int32)), name="reward"),
        ...           Parameter(Tensor(np.arange(100 * 4).reshape(100, 4).astype(np.float32)), name="state_")]
        >>> buffer_get = ops.BufferGetItem(capacity, shapes, types)
        >>> output = buffer_get(buffer, count, head, index)
        >>> print(output)
            (Tensor(shape=[4], dtype=Float32, value=
                [ 1.20000000e+01, 1.30000000e+01, 1.40000000e+01, 1.50000000e+01]),
             Tensor(shape=[2], dtype=Int32, value= [6, 7]),
             Tensor(shape=[1], dtype=Int32, value= [1]),
             Tensor(shape=[4], dtype=Float32, value=
                [ 1.20000000e+01, 1.30000000e+01, 1.40000000e+01, 1.50000000e+01]))

    """
    @prim_attr_register
    def __init__(self, capacity, buffer_shape, buffer_dtype):
        """Initialize BufferGetItem."""
        self.init_prim_io_names(inputs=["buffer"], outputs=["item"])
        validator.check_int(capacity, 1, validator.GE, "capacity", self.name)
        self._buffer_shape = buffer_shape
        self._buffer_dtype = buffer_dtype
        self._n = len(buffer_shape)
        buffer_elements = []
        for shape in buffer_shape:
            buffer_elements.append(reduce(lambda x, y: x * y, shape))
        self.add_prim_attr('buffer_elements', buffer_elements)
        self.add_prim_attr('buffer_dtype', buffer_dtype)
        self.add_prim_attr('capacity', capacity)
        self.add_prim_attr('side_effect_mem', True)
        if context.get_context('device_target') == "Ascend":
            self.add_prim_attr('device_target', "CPU")

    def infer_shape(self, data_shape, count_shape, head_shape, index_shape):
        validator.check_value_type("shape of data", data_shape, [tuple, list], self.name)
        return tuple(self._buffer_shape)

    def infer_dtype(self, data_type, count_type, head_type, index_type):
        validator.check_type_name("count type", count_type, (mstype.int32), self.name)
        validator.check_type_name("head type", head_type, (mstype.int32), self.name)
        validator.check_type_name("index type", index_type, (mstype.int64, mstype.int32), self.name)
        return tuple(self._buffer_dtype)
