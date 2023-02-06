# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
"""Operations for sequence"""

from mindspore.ops.primitive import Primitive, prim_attr_register


class ListAppend(Primitive):
    r"""
    Append element to the end of list.

    .. note::
        This operation is used for dynamic length list and this it is only for internal used.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **input_data** (List) - The list for target to append. Must be dynamic length sequence
        - **target** (Any Object) - The target element to be appended. The shape and type of target must be the same as
          as the element within 'input_data'.

    Outputs:
        Dynamic length list after append.

    Raises:
        TypeError: The 'input_data' is not dynamic length list.
        ValueError: The shape or type of 'target' is not the same as the element within 'input_data'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize ListAppend"""
        self.init_prim_io_names(inputs=['input_data', 'target'], outputs=['output_data'])


class SequenceSlice(Primitive):
    r"""
    Sequence slice operation.

    .. note::
        This it is only for internal used. The sequence input should be dynamic length sequence or at least one of
        start/end/step should be variable.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **seq** (Union[List, Tuple]) - The sequence to slice.
        - **start** (int) - start index of slice.
        - **stop** (int) - stop index of slice.
        - **step** (int) - step of slice.

    Outputs:
        Dynamic length sequence after slice.

    Raises:
        TypeError: The 'seq' input is neither list or tuple.
        ValueError: The 'seq' input is not dynamic length and none of start/end/step is variable.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize SequenceSlice"""
        self.init_prim_io_names(inputs=['seq', 'start', 'stop', 'step'], outputs=['output_data'])


class SequenceSliceSetItem(Primitive):
    r"""
    Sequence slice setitem operation.

    .. note::
        This it is only for internal used. The sequence input should be dynamic length sequence or at least one of
        start/end/step should be variable.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **seq** (Union[List, Tuple]) - The sequence to perform slice setitem.
        - **target** (Union[List, Tuple]) - The target item to set.
        - **start** (int) - start index of slice.
        - **stop** (int) - stop index of slice.
        - **step** (int) - step of slice.

    Outputs:
        Dynamic length sequence after slice setitem.

    Raises:
        ValueError: The 'seq' and 'target' input is not dynamic length and none of start/end/step is variable.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize SequenceSliceSetItem"""
        self.init_prim_io_names(inputs=['seq', 'target', 'start', 'stop', 'step'], outputs=['output_data'])


class SequenceSliceGrad(Primitive):
    """Reverse of slice."""

    @prim_attr_register
    def __init__(self):
        """Initialize SequenceSliceGrad"""
        self.init_prim_io_names(inputs=['dy', 'x', 'start', 'stop', 'step'], outputs=['dx'])


class SequenceAdd(Primitive):
    r"""
    Add elements of two sequence together.

    .. note::
        This it is only for internal used. At least one of the sequence should be dynamic length sequence.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **input_1** (Union[List, Tuple]) - The first sequence for sequence addition.
        - **input_2** (Union[List, Tuple]) - The second sequence for sequence addition.

    Outputs:
        Dynamic length sequence after addition.

    Raises:
        TypeError: The 'input_1' and 'input_2' are not both list or tuple.
        TypeError: Both of 'input_1' and 'input_2' are not dynamic length sequence.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize SequenceAdd"""
        self.init_prim_io_names(inputs=['input_1', 'input_2'], outputs=['output_data'])


class SequenceAddOffset(Primitive):
    r"""
    Get offsets of SequenceAdd inputs within its output, refs to ConcatOffset.

    .. note::
        This it is only for internal used. At least one of the sequence should be dynamic length sequence.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **input_0** (List, Tuple) - A Tuple/List.
        - **input_1** (List, Tuple) - A Tuple/List.

    Outputs:
        A tuple of offsets of SequenceAdd inputs within its output.

    Raises:
        TypeError: The data in 'input_0' and 'input_1' are not both list or tuple.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize SequenceAddOffset"""
        self.init_prim_io_names(inputs=['shape_0', 'shape_1'], outputs=['output'])


class TupleToTensor(Primitive):
    r"""
    Convert tuple to tensor

    If the type of the first number in the tuple is integer, the data type of the output tensor is int.
    Otherwise, the data type of the output tensor is float.

    .. note::
        This it is only for internal used. The input_tuple can be constant/variable value or dynamic length tuple.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **input_tuple** (Tuple) - A tuple of numbers. These numbers have the same type. The shape is :math:`(N,)`
        - **dtype** (mindspore.dtype): The target data type. Default: mindspore.float32.

    Outputs:
        Tensor, if the input tuple contains `N` numbers, then the shape of the output tensor is (N,).

    Raises:
        TypeError: If `input_tuple` is not a tuple.
        ValueError: If length of `input_tuple` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize TupleToTensor"""
        self.init_prim_io_names(inputs=['input_tuple', 'dtype'], outputs=['output_data'])


class ListToTensor(Primitive):
    r"""
    Convert list to tensor

    If the type of the first number in the list is integer, the data type of the output tensor is int.
    Otherwise, the data type of the output tensor is float.

    .. note::
        This it is only for internal used. The input_list can be constant/variable value or dynamic length list.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **input_list** (List) - A tuple of numbers. These numbers have the same type. The shape is :math:`(N,)`
        - **dtype** (mindspore.dtype): The target data type. Default: mindspore.float32.

    Outputs:
        Tensor, if the input tuple contains `N` numbers, then the shape of the output tensor is (N,).

    Raises:
        TypeError: If `input_list` is not a list.
        ValueError: If length of `input_list` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize ListToTensor"""
        self.init_prim_io_names(inputs=['input_list', 'dtype'], outputs=['output_data'])


class TensorToTuple(Primitive):
    r"""
    Convert tensor to tuple

    .. note::
        This it is only for internal used. The input_tensor can be constant/variable value.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **input_tensor** (Tensor) - The input_tensor must be a 1-D Tensor.

    Outputs:
        Tuple, the length is equal to tensor shape.

    Raises:
        TypeError: If `input_tensor` is not a tensor.
        ValueError: If rank of `input_tensor` is not 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize TensorToTuple"""
        self.init_prim_io_names(inputs=['input_tensor'], outputs=['output_data'])


class TensorToList(Primitive):
    r"""
    Convert tensor to list

    .. note::
        This it is only for internal used. The input_tensor can be constant/variable value.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **input_tensor** (Tensor) - The input_tensor must be a 1-D Tensor.

    Outputs:
        List, the length is equal to tensor shape.

    Raises:
        TypeError: If `input_tensor` is not a tensor.
        ValueError: If rank of `input_tensor` is not 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize TensorToList"""
        self.init_prim_io_names(inputs=['input_tensor'], outputs=['output_data'])


class TensorToScalar(Primitive):
    r"""
    Convert tensor to scalar

    .. note::
        This it is only for internal used. The input_tensor can be constant/variable value.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **input_tensor** (Tensor) - The input_tensor must be a 0-D Tensor.

    Outputs:
        Scalar, a constant or variable scalar.

    Raises:
        TypeError: If `input_tensor` is not a tensor.
        ValueError: If rank of `input_tensor` is not 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize TensorToScalar"""
        self.init_prim_io_names(inputs=['input_tensor'], outputs=['output_data'])


class SequenceCount(Primitive):
    r"""
    Support sequence count operation 'seq.count(target)'.

    .. note::
        This it is only for internal used.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **sequence** (Union[List, Tuple]) - The sequence to count elements.
        - **target** (Any Object) - The target element to find in sequence.

    Outputs:
        Integer, counting of target element.

    Raises:
        TypeError: The 'sequence' is not list or tuple.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize SequenceCount"""
        self.init_prim_io_names(inputs=['sequence', 'target'], outputs=['output_data'])


class SequenceIndex(Primitive):
    r"""
    Support sequence index operation 'seq.index(target)'.

    .. note::
        This it is only for internal used.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **sequence** (Union[List, Tuple]) - The sequence to find the index value of the first occurrence.
        - **target** (Any Object) - The target element to find in sequence.

    Outputs:
        Integer, the index value of the first occurrence of the target element.

    Raises:
        TypeError: The 'sequence' is not list or tuple.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize SequenceIndex"""
        self.init_prim_io_names(inputs=['sequence', 'target'], outputs=['output_data'])


class SequenceMul(Primitive):
    r"""
    Support sequence multiplication operation 'seq.mul(scalar)'.

    .. note::
        This it is only for internal used.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **sequence** (Union[List, Tuple]) - The sequence to count elements.
        - **scalar** (Any Object) - The times to replicate the sequence.

    Outputs:
        List or tuple with 'scalar' times multiplication.

    Raises:
        TypeError: The 'sequence' is not list or tuple.
        ValueError: Both 'sequence' and 'scalar' is constant.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize SequenceMul"""
        self.init_prim_io_names(inputs=['sequence', 'scalar'], outputs=['output_data'])


class SequenceZerosLike(Primitive):
    r"""
    Returns a Sequence with a value of 0 and its shape and data type is the same as the input.

    .. note::
        This it is only for internal used.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **sequence** (Union[List, Tuple]) - The sequence to count elements.

    Outputs:
        List or tuple filled with zeros.

    Raises:
        TypeError: The 'sequence' is not list or tuple.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize SequenceZerosLike"""
        self.init_prim_io_names(inputs=['sequence'], outputs=['output_data'])


class make_range(Primitive):
    r"""
    Creates a sequence of numbers in range [start, limit) with step size delta.

    .. note::
        This it is only for internal used.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **start** (Union[int, float]) - Start of interval.
        - **limit** (Union[int, float]) - End of interval.
        - **delta** (Union[int, float]) - Spacing between values.

    Outputs:
        A 1-D Sequence, with the same type as the inputs.

    Raises:
        TypeError: If datatype of `start`, `limit` or `delta` is not same.
        TypeError: If datatype of `start`, `limit` or `delta` is not supported.
        ValueError: If `delta` = 0.
        ValueError: If `start` >= `limit` when `delta` > 0.
        ValueError: If `start` <= `limit` when `delta` < 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize make_range"""
        self.init_prim_io_names(inputs=['start', 'limit', 'delta'], outputs=['output_data'])


class sequence_len(Primitive):
    r"""
    Returns length of Sequence.

    .. note::
        This it is only for internal used.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **sequence** (Union[List, Tuple]) - The sequence.

    Outputs:
        Integer, length of Sequence.

    Raises:
        TypeError: The 'sequence' is not list or tuple.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize sequence_len"""
        self.init_prim_io_names(inputs=['sequence'], outputs=['output_data'])
