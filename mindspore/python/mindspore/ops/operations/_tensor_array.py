# Copyright 2021 Huawei Technologies Co., Ltd
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

"""Operators for TensorArray."""

import mindspore as ms
from mindspore import _checkparam as validator
from ...common import dtype as mstype
from ..primitive import prim_attr_register, PrimitiveWithInfer, Primitive


class TensorArray(PrimitiveWithInfer):
    r"""
    TensorArrayCreate used to create a TensorArray and return an unique handle.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        dtype (mindspore.dtype): the data type in the TensorArray.
        element_shape (tuple[int]): the shape of each tensor in a TensorArray.
        dynamic_size (bool): If true the TensorArray can increase the size. Default: ``True``.
        size (int): The size of the TensorArray if dynamic_size = False.
        name (string): the name of this TensorArray. Default: "TA".

    Inputs:
        None.

    Outputs:
        - **output** (Tensor[mindspore.int64]) - an unique handle binded to the TensorArray.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.ops as ops
        >>> create_op = ops.TensorArray(mindspore.int32, ())
        >>> handle = create_op()
        >>> print(handle)
        0
    """
    @prim_attr_register
    def __init__(self, dtype, element_shape, dynamic_size=True, size=0, name="TA"):
        validator.check_type_name("dtype", dtype, mstype.number_type + (mstype.bool_,), self.name)
        validator.check_int(size, 0, validator.GE, "size", self.name)
        self.add_prim_attr('dtype', dtype)
        self.add_prim_attr('element_shape', element_shape)
        self.add_prim_attr('dynamic_size', dynamic_size)
        self.add_prim_attr('size', size)
        self.add_prim_attr('side_effect_mem', True)
        self.add_prim_attr('name', name)

    def infer_shape(self):
        return ()

    def infer_dtype(self):
        return mstype.int64


class TensorArrayWrite(PrimitiveWithInfer):
    r"""
    TensorArrayWrite used to write tensor into a created TensorArray.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Inputs:
        - **index** (Tensor[int64]) - The position to write.
        - **value** (Tensor) - The value to add into the TensorArray.
        - **handle** (Tensor[int64]) - The handle pointed to the TensorArray.

    Outputs:
        None.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.ops as ops
        >>> create_op = ops.TensorArray(mindspore.int32, ())
        >>> handle = create_op()
        >>> write_op = ops.TensorArrayWrite()
        >>> write_op.write(handle, 0, 1)
    """
    @prim_attr_register
    def __init__(self):
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, handle_shape, index_shape, value_shape):
        return ()

    def infer_dtype(self, handle_type, index_type, value_type):
        validator.check_type_name("handle", handle_type, (ms.int64), self.name)
        validator.check_type_name("index", index_type, (int, ms.int64), self.name)
        validator.check_type_name("value", value_type, mstype.number_type + (mstype.bool_,), self.name)
        return mstype.int64


class TensorArrayRead(PrimitiveWithInfer):
    r"""
    TensorArrayRead used to read tensor from a created TensorArray by the given index.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        dtype (mindspore.dtype): the data type in the TensorArray.
        element_shape (tuple[int]): the shape of each tensor in a TensorArray.

    Inputs:
        - **index** (Tensor[int64]) - The position to read.
        - **handle** (mindspore.int64) - The handle pointed to the TensorArray.

    Outputs:
        - **output** (Tensor) - the value in position index.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.ops as ops
        >>> create_op = ops.TensorArray(mindspore.int32, ())
        >>> handle = create_op()
        >>> write_op = ops.TensorArrayWrite()
        >>> write_op.write(handle, 0, 1)
        >>> read_op = ops.TensorArrayRead(mindspore.int32, ())
        >>> ans = read_op(handle, 0)
        >>> print(ans)
        1
    """
    @prim_attr_register
    def __init__(self, dtype, element_shape):
        validator.check_type_name("dtype", dtype, mstype.number_type + (mstype.bool_,), self.name)
        self.add_prim_attr('dtype', dtype)
        self.add_prim_attr('element_shape', element_shape)
        self.add_prim_attr('side_effect_mem', True)
        self.dtype = dtype
        self.shape = element_shape

    def infer_shape(self, handle_shape, index_shape):
        return self.shape

    def infer_dtype(self, handle_type, index_type):
        validator.check_type_name("handle", handle_type, (ms.int64), self.name)
        validator.check_type_name("index", index_type, (int, ms.int64), self.name)
        return self.dtype


class TensorArrayClose(PrimitiveWithInfer):
    r"""
    TensorArrayClose used to close the created TensorArray. The resources in TensorArray will be deleted.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Inputs:
        - **handle** (mindspore.int64) - The handle pointed to the TensorArray.

    Outputs:
        None.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.ops as ops
        >>> create_op = ops.TensorArray(mindspore.int32, ())
        >>> handle = create_op()
        >>> close_op = ops.TensorArrayClose()
        >>> close_op(handle)
    """
    @prim_attr_register
    def __init__(self):
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, handle_shape):
        return ()

    def infer_dtype(self, handle_type):
        validator.check_type_name("handle", handle_type, (ms.int64), self.name)
        return mstype.int64


class TensorArrayClear(PrimitiveWithInfer):
    r"""
    TensorArrayClear used to reset the created TensorArray. The instance of TensorArray is still aviliable.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Inputs:
        - **handle** (mindspore.int64) - The handle pointed to the TensorArray.

    Outputs:
        None.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.ops as ops
        >>> create_op = ops.TensorArray(mindspore.int32, ())
        >>> handle = create_op()
        >>> clear_op = ops.TensorArrayClear()
        >>> clear_op(handle)
    """
    @prim_attr_register
    def __init__(self):
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, handle_shape):
        return ()

    def infer_dtype(self, handle_type):
        validator.check_type_name("handle", handle_type, (ms.int64), self.name)
        return mstype.int64


class TensorArrayStack(Primitive):
    r"""
    TensorArrayStack used to stack the tensors in a created TensorArray into one tensor.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        dtype (mindspore.dtype): the data type in the TensorArray.
        element_shape (tuple[int]): the shape of each tensor in a TensorArray.

    Inputs:
        - **handle** (mindspore.int64) - The handle pointed to the TensorArray.

    Outputs:
        - **output** (Tensor) - the stacked value from the TensorArray.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.ops as ops
        >>> create_op = ops.TensorArray(mindspore.int32, ())
        >>> handle = create_op()
        >>> write_op = ops.TensorArrayWrite()
        >>> write_op.write(handle, 0, 1)
        >>> write_op.write(handle, 1, 2)
        >>> stack_op = ops.TensorArrayStack(mindspore.int32, ())
        >>> ans = stack_op(handle)
        >>> print(ans)
        [1 2]
    """
    @prim_attr_register
    def __init__(self, dtype, element_shape, dynamic_size, size):
        """Initialize TensorArrayStack"""
        self.init_prim_io_names(inputs=[''], outputs=['output'])
        self.add_prim_attr('dtype', dtype)
        self.add_prim_attr('element_shape', element_shape)
        self.add_prim_attr('is_dynamic_shape', dynamic_size)
        self.add_prim_attr('size', size)
        self.add_prim_attr('side_effect_mem', True)


class TensorArraySize(PrimitiveWithInfer):
    r"""
    TensorArraySize used to get the logical size of the created TensorArray.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Inputs:
        - **handle** (mindspore.int64) - The handle pointed to the TensorArray.

    Outputs:
        - **output** (Tensor[mindspore.int64]) - the logical size of the TensorArray.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.ops as ops
        >>> create_op = ops.TensorArray(mindspore.int32, ())
        >>> handle = create_op()
        >>> size_op = ops.TensorArraySize()
        >>> size = size_op(handle)
    """
    @prim_attr_register
    def __init__(self):
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, handle_shape):
        return ()

    def infer_dtype(self, handle_type):
        validator.check_type_name("handle", handle_type, (ms.int64), self.name)
        return mstype.int64


class TensorArrayGather(PrimitiveWithInfer):
    r"""
    TensorArrayGather used to gather specified elements from the created TensorArray.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        dtype (mindspore.dtype): the data type in the TensorArray.
        element_shape (tuple[int]): the shape of each tensor in a TensorArray.

    Inputs:
        - **handle** (mindspore.int64) - The handle pointed to the TensorArray.
        - **indices** (mindspore.int32) - The locations of the gathered elements.

    Outputs:
        - **output** (Tensor) - The gathered value from the TensorArray.

    Examples:
        >>> import mindspore
        >>> import mindspore.ops as ops
        >>> from mindspore import numpy as mnp
        >>> create_op = ops.TensorArray(mindspore.float32, dynamic_size=False, element_shape=(8,))
        >>> handle = create_op()
        >>> indices = mnp.range(0, 25, 1, mindspore.int32)
        >>> gather_op = ops.TensorArrayGather(dtype=mindspore.float32, element_shape=(8,))
        >>> gather_result = gather_op(handle, indices)
    """
    @prim_attr_register
    def __init__(self, dtype, element_shape):
        self.init_prim_io_names(inputs=['handle', 'indices'], outputs=['value'])
        self.add_prim_attr("side_effect_mem", True)
        self.dtype = dtype
        self.element_shape = element_shape

    def infer_shape(self, handle, indices):
        if len(indices) != 1:
            return ValueError("indices dimension should be equal to 1")
        return [indices[0]] + list(self.element_shape)

    def infer_dtype(self, handle, indices):
        validator.check_type_name("handle", handle, (ms.int64), self.name)
        validator.check_type_name("indices", indices, (ms.int32), self.name)
        return self.dtype
