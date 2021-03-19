# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
The module transforms.c_transforms provides common operations, including OneHotOp and TypeCast.
"""
from enum import IntEnum
import numpy as np

import mindspore.common.dtype as mstype
import mindspore._c_dataengine as cde

from .validators import check_num_classes, check_de_type, check_fill_value, check_slice_option, check_slice_op, \
    check_mask_op, check_pad_end, check_concat_type, check_random_transform_ops
from ..core.datatypes import mstype_to_detype


class TensorOperation:
    """
    Base class Tensor Ops
    """
    def __call__(self, *input_tensor_list):
        tensor_row = []
        for tensor in input_tensor_list:
            try:
                tensor_row.append(cde.Tensor(np.asarray(tensor)))
            except RuntimeError:
                raise TypeError("Invalid user input. Got {}: {}, cannot be converted into tensor." \
                      .format(type(tensor), tensor))
        callable_op = cde.Execute(self.parse())
        output_tensor_list = callable_op(tensor_row)
        for i, element in enumerate(output_tensor_list):
            arr = element.as_array()
            if arr.dtype.char == 'S':
                output_tensor_list[i] = np.char.decode(arr)
            else:
                output_tensor_list[i] = arr
        return output_tensor_list[0] if len(output_tensor_list) == 1 else tuple(output_tensor_list)

    def parse(self):
        raise NotImplementedError("TensorOperation has to implement parse() method.")


class OneHot(cde.OneHotOp):
    """
    Tensor operation to apply one hot encoding.

    Args:
        num_classes (int): Number of classes of objects in dataset.
            It should be larger than the largest label number in the dataset.

    Raises:
        RuntimeError: feature size is bigger than num_classes.

    Examples:
        >>> # Assume that dataset has 10 classes, thus the label ranges from 0 to 9
        >>> onehot_op = c_transforms.OneHot(num_classes=10)
        >>> mnist_dataset = mnist_dataset.map(operations=onehot_op, input_columns=["label"])
    """

    @check_num_classes
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__(num_classes)


class Fill(cde.FillOp):
    """
    Tensor operation to fill all elements in the tensor with the specified value.
    The output tensor will have the same shape and type as the input tensor.

    Args:
        fill_value (Union[str, bytes, int, float, bool])) : scalar value
            to fill the tensor with.

    Examples:
        >>> import numpy as np
        >>> # generate a 1D integer numpy array from 0 to 4
        >>> def generator_1d():
        ...     for i in range(5):
        ...         yield (np.array([i]),)
        >>> generator_dataset = ds.GeneratorDataset(generator_1d, column_names="col1")
        >>> # [[0], [1], [2], [3], [4]]
        >>> fill_op = c_transforms.Fill(3)
        >>> generator_dataset = generator_dataset.map(operations=fill_op)
        >>> # [[3], [3], [3], [3], [3]]
    """

    @check_fill_value
    def __init__(self, fill_value):
        super().__init__(cde.Tensor(np.array(fill_value)))


class TypeCast(cde.TypeCastOp):
    """
    Tensor operation to cast to a given MindSpore data type.

    Args:
        data_type (mindspore.dtype): mindspore.dtype to be cast to.

    Examples:
        >>> import numpy as np
        >>> import mindspore.common.dtype as mstype
        >>>
        >>> # Generate 1d int numpy array from 0 - 63
        >>> def generator_1d():
        ...     for i in range(64):
        ...         yield (np.array([i]),)
        >>>
        >>> dataset = ds.GeneratorDataset(generator_1d, column_names='col')
        >>> type_cast_op = c_transforms.TypeCast(mstype.int32)
        >>> dataset = dataset.map(operations=type_cast_op)
    """

    @check_de_type
    def __init__(self, data_type):
        data_type = mstype_to_detype(data_type)
        self.data_type = str(data_type)
        super().__init__(data_type)


class _SliceOption(cde.SliceOption):
    """
    Internal class SliceOption to be used with SliceOperation

    Args:
        _SliceOption(Union[int, list(int), slice, None, Ellipsis, bool, _SliceOption]):

            1.  :py:obj:`int`: Slice this index only along the dimension. Negative index is supported.
            2.  :py:obj:`list(int)`: Slice these indices along the dimension. Negative indices are supported.
            3.  :py:obj:`slice`: Slice the generated indices from the slice object along the dimension.
            4.  :py:obj:`None`: Slice the whole dimension. Similar to :py:obj:`:` in Python indexing.
            5.  :py:obj:`Ellipsis`: Slice the whole dimension. Similar to :py:obj:`:` in Python indexing.
            6.  :py:obj:`boolean`: Slice the whole dimension. Similar to :py:obj:`:` in Python indexing.
    """

    @check_slice_option
    def __init__(self, slice_option):
        if isinstance(slice_option, int) and not isinstance(slice_option, bool):
            slice_option = [slice_option]
        elif slice_option is Ellipsis:
            slice_option = True
        elif slice_option is None:
            slice_option = True
        super().__init__(slice_option)


class Slice(cde.SliceOp):
    """
    Slice operation to extract a tensor out using the given n slices.

    The functionality of Slice is similar to NumPy's indexing feature.
    (Currently only rank-1 tensors are supported).

    Args:
        slices (Union[int, list[int], slice, None, Ellipsis]):
            Maximum `n` number of arguments to slice a tensor of rank `n` .
            One object in slices can be one of:

            1.  :py:obj:`int`: Slice this index only along the first dimension. Negative index is supported.
            2.  :py:obj:`list(int)`: Slice these indices along the first dimension. Negative indices are supported.
            3.  :py:obj:`slice`: Slice the generated indices from the slice object along the first dimension.
                Similar to start:stop:step.
            4.  :py:obj:`None`: Slice the whole dimension. Similar to :py:obj:`[:]` in Python indexing.
            5.  :py:obj:`Ellipsis`: Slice the whole dimension, same result with `None`.

    Examples:
        >>> # Data before
        >>> # |   col   |
        >>> # +---------+
        >>> # | [1,2,3] |
        >>> # +---------|
        >>> data = [[1, 2, 3]]
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["col"])
        >>> # slice indices 1 and 2 only
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=c_transforms.Slice(slice(1,3)))
        >>> # Data after
        >>> # |   col   |
        >>> # +---------+
        >>> # |  [2,3]  |
        >>> # +---------|
    """

    @check_slice_op
    def __init__(self, *slices):
        slice_input_ = list(slices)
        slice_input_ = [_SliceOption(slice_dim) for slice_dim in slice_input_]
        super().__init__(slice_input_)


class Relational(IntEnum):
    EQ = 0
    NE = 1
    GT = 2
    GE = 3
    LT = 4
    LE = 5


DE_C_RELATIONAL = {Relational.EQ: cde.RelationalOp.EQ,
                   Relational.NE: cde.RelationalOp.NE,
                   Relational.GT: cde.RelationalOp.GT,
                   Relational.GE: cde.RelationalOp.GE,
                   Relational.LT: cde.RelationalOp.LT,
                   Relational.LE: cde.RelationalOp.LE}


class Mask(cde.MaskOp):
    """
    Mask content of the input tensor with the given predicate.
    Any element of the tensor that matches the predicate will be evaluated to True, otherwise False.

    Args:
        operator (Relational): One of the relational operators EQ, NE LT, GT, LE or GE
        constant (Union[str, int, float, bool]): Constant to be compared to.
            Constant will be cast to the type of the input tensor.
        dtype (mindspore.dtype, optional): Type of the generated mask (Default to bool).

    Examples:
        >>> from mindspore.dataset.transforms.c_transforms import Relational
        >>> # Data before
        >>> # |  col   |
        >>> # +---------+
        >>> # | [1,2,3] |
        >>> # +---------+
        >>> data = [[1, 2, 3]]
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["col"])
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=c_transforms.Mask(Relational.EQ, 2))
        >>> # Data after
        >>> # |       col         |
        >>> # +--------------------+
        >>> # | [False,True,False] |
        >>> # +--------------------+
    """

    @check_mask_op
    def __init__(self, operator, constant, dtype=mstype.bool_):
        dtype = mstype_to_detype(dtype)
        constant = cde.Tensor(np.array(constant))
        super().__init__(DE_C_RELATIONAL[operator], constant, dtype)


class PadEnd(cde.PadEndOp):
    """
    Pad input tensor according to pad_shape, need to have same rank.

    Args:
        pad_shape (list(int)): List of integers representing the shape needed. Dimensions that set to `None` will
            not be padded (i.e., original dim will be used). Shorter dimensions will truncate the values.
        pad_value (Union[str, bytes, int, float, bool]), optional): Value used to pad. Default to 0 or empty
            string in case of tensors of strings.

    Examples:
        >>> # Data before
        >>> # |   col   |
        >>> # +---------+
        >>> # | [1,2,3] |
        >>> # +---------|
        >>> data = [[1, 2, 3]]
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["col"])
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=c_transforms.PadEnd(pad_shape=[4],
        ...                                                                                pad_value=10))
        >>> # Data after
        >>> # |    col     |
        >>> # +------------+
        >>> # | [1,2,3,10] |
        >>> # +------------|
    """

    @check_pad_end
    def __init__(self, pad_shape, pad_value=None):
        if pad_value is not None:
            pad_value = cde.Tensor(np.array(pad_value))
        super().__init__(cde.TensorShape(pad_shape), pad_value)


class Concatenate(cde.ConcatenateOp):
    """
    Tensor operation that concatenates all columns into a single tensor.

    Args:
        axis (int, optional): Concatenate the tensors along given axis (Default=0).
        prepend (numpy.array, optional): NumPy array to be prepended to the already concatenated tensors (Default=None).
        append (numpy.array, optional): NumPy array to be appended to the already concatenated tensors (Default=None).

    Examples:
        >>> import numpy as np
        >>> # concatenate string
        >>> prepend_tensor = np.array(["dw", "df"], dtype='S')
        >>> append_tensor = np.array(["dwsdf", "df"], dtype='S')
        >>> concatenate_op = c_transforms.Concatenate(0, prepend_tensor, append_tensor)
        >>> data = [["This","is","a","string"]]
        >>> dataset = ds.NumpySlicesDataset(data)
        >>> dataset = dataset.map(operations=concatenate_op)
    """

    @check_concat_type
    def __init__(self, axis=0, prepend=None, append=None):
        if prepend is not None:
            prepend = cde.Tensor(np.array(prepend))
        if append is not None:
            append = cde.Tensor(np.array(append))
        super().__init__(axis, prepend, append)


class Duplicate(cde.DuplicateOp):
    """
    Duplicate the input tensor to output, only support transform one column each time.

    Examples:
        >>> # Data before
        >>> # |  x      |
        >>> # +---------+
        >>> # | [1,2,3] |
        >>> # +---------+
        >>> data = [[1,2,3]]
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["x"])
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=c_transforms.Duplicate(),
        ...                                                 input_columns=["x"],
        ...                                                 output_columns=["x", "y"],
        ...                                                 column_order=["x", "y"])
        >>> # Data after
        >>> # |  x      |  y      |
        >>> # +---------+---------+
        >>> # | [1,2,3] | [1,2,3] |
        >>> # +---------+---------+
    """


class Unique(cde.UniqueOp):
    """
    Perform the unique operation on the input tensor, only support transform one column each time.

    Return 3 tensor: unique output tensor, index tensor, count tensor.

    Unique output tensor contains all the unique elements of the input tensor
    in the same order that they occur in the input tensor.

    Index tensor that contains the index of each element of the input tensor in the unique output tensor.

    Count tensor that contains the count of each element of the output tensor in the input tensor.

    Note:
        Call batch op before calling this function.

    Examples:
        >>> # Data before
        >>> # |  x                 |
        >>> # +--------------------+
        >>> # | [[0,1,2], [1,2,3]] |
        >>> # +--------------------+
        >>> data = [[[0,1,2], [1,2,3]]]
        >>> dataset = ds.NumpySlicesDataset(data, ["x"])
        >>> dataset = dataset.map(operations=c_transforms.Unique(),
        ...                       input_columns=["x"],
        ...                       output_columns=["x", "y", "z"],
        ...                       column_order=["x", "y", "z"])
        >>> # Data after
        >>> # |  x      |  y              |z        |
        >>> # +---------+-----------------+---------+
        >>> # | [0,1,2,3] | [0,1,2,1,2,3] | [1,2,2,1]
        >>> # +---------+-----------------+---------+

    """


class Compose():
    """
    Compose a list of transforms into a single transform.

    Args:
        transforms (list): List of transformations to be applied.

    Examples:
        >>> compose = c_transforms.Compose([c_vision.Decode(), c_vision.RandomCrop(512)])
        >>> image_folder_dataset = image_folder_dataset.map(operations=compose)
    """

    @check_random_transform_ops
    def __init__(self, transforms):
        self.transforms = transforms

    def parse(self):
        operations = []
        for op in self.transforms:
            if op and getattr(op, 'parse', None):
                operations.append(op.parse())
            else:
                operations.append(op)
        return cde.ComposeOperation(operations)


class RandomApply():
    """
    Randomly perform a series of transforms with a given probability.

    Args:
        transforms (list): List of transformations to be applied.
        prob (float, optional): The probability to apply the transformation list (default=0.5)

    Examples:
        >>> rand_apply = c_transforms.RandomApply([c_vision.RandomCrop(512)])
        >>> image_folder_dataset = image_folder_dataset.map(operations=rand_apply)
    """

    @check_random_transform_ops
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob = prob

    def parse(self):
        operations = []
        for op in self.transforms:
            if op and getattr(op, 'parse', None):
                operations.append(op.parse())
            else:
                operations.append(op)
        return cde.RandomApplyOperation(self.prob, operations)


class RandomChoice():
    """
    Randomly select one transform from a list of transforms to perform operation.

    Args:
        transforms (list): List of transformations to be chosen from to apply.

    Examples:
        >>> rand_choice = c_transforms.RandomChoice([c_vision.CenterCrop(50), c_vision.RandomCrop(512)])
        >>> image_folder_dataset = image_folder_dataset.map(operations=rand_choice)
    """

    @check_random_transform_ops
    def __init__(self, transforms):
        self.transforms = transforms

    def parse(self):
        operations = []
        for op in self.transforms:
            if op and getattr(op, 'parse', None):
                operations.append(op.parse())
            else:
                operations.append(op)
        return cde.RandomChoiceOperation(operations)
