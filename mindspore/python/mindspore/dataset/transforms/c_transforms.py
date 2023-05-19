# Copyright 2019-2022 Huawei Technologies Co., Ltd
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

from mindspore.common import dtype as mstype
import mindspore._c_dataengine as cde

from .validators import check_num_classes, check_ms_type, check_fill_value, check_slice_option, check_slice_op, \
    check_mask_op, check_pad_end, check_concat_type, check_random_transform_ops, check_plugin, deprecated_c_transforms
from ..core.datatypes import mstype_to_detype


# pylint: disable=super-init-not-called
class TensorOperation:
    """
    Base class Tensor Ops
    """

    def __init__(self):
        self.callable_op_ = None

    def __call__(self, *input_tensor_list):
        tensor_row = []
        for tensor in input_tensor_list:
            try:
                tensor_row.append(cde.Tensor(np.asarray(tensor)))
            except RuntimeError:
                raise TypeError("Invalid user input. Got {}: {}, cannot be converted into tensor." \
                                .format(type(tensor), tensor))
        if not hasattr(self, 'callable_op_') or self.callable_op_ is None:
            self.callable_op_ = cde.Execute(self.parse())
        output_tensor_list = self.callable_op_(tensor_row)
        for i, element in enumerate(output_tensor_list):
            arr = element.as_array()
            if arr.dtype.char == 'S':
                output_tensor_list[i] = np.char.decode(arr)
            else:
                output_tensor_list[i] = arr
        return output_tensor_list[0] if len(output_tensor_list) == 1 else tuple(output_tensor_list)

    def parse(self):
        """parse function - not yet implemented"""
        raise NotImplementedError("TensorOperation has to implement parse() method.")


class OneHot(TensorOperation):
    """
    Tensor operation to apply one hot encoding.

    Args:
        num_classes (int): Number of classes of objects in dataset.
            It should be larger than the largest label number in the dataset.

    Raises:
        TypeError: `num_classes` is not of type int.
        RuntimeError: Input tensor is not of type int.
        RuntimeError: Input tensor is not a 1-D tensor.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> # Assume that dataset has 10 classes, thus the label ranges from 0 to 9
        >>> onehot_op = c_transforms.OneHot(num_classes=10)
        >>> mnist_dataset = mnist_dataset.map(operations=onehot_op, input_columns=["label"])
    """

    @deprecated_c_transforms()
    @check_num_classes
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def parse(self):
        # Use smoothing_rate=0 for legacy support for OneHot
        return cde.OneHotOperation(self.num_classes, 0)


class Fill(TensorOperation):
    """
    Tensor operation to fill all elements in the tensor with the specified value.
    The output tensor will have the same shape and type as the input tensor.

    Args:
        fill_value (Union[str, bytes, int, float, bool]) : scalar value
            to fill the tensor with.

    Raises:
        TypeError: If `fill_value` is not of type str, float, bool, int or bytes.

    Supported Platforms:
        ``CPU``

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

    @deprecated_c_transforms()
    @check_fill_value
    def __init__(self, fill_value):
        self.fill_value = cde.Tensor(np.array(fill_value))

    def parse(self):
        return cde.FillOperation(self.fill_value)


class TypeCast(TensorOperation):
    """
    Tensor operation to cast to a given MindSpore data type.

    Note:
        This operation supports running on Ascend or GPU platforms by Offload.

    Args:
        data_type (mindspore.dtype): mindspore.dtype to be cast to.

    Raises:
        TypeError: If `data_type` is not of type bool, int, float or string.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import dtype as mstype
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

    @deprecated_c_transforms()
    @check_ms_type
    def __init__(self, data_type):
        data_type = mstype_to_detype(data_type)
        self.data_type = str(data_type)

    def parse(self):
        return cde.TypeCastOperation(self.data_type)


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


class Slice(TensorOperation):
    """
    Slice operation to extract a tensor out using the given n slices.

    The functionality of Slice is similar to NumPy's indexing feature (Currently only rank-1 tensors are supported).

    Args:
        slices (Union[int, list[int], slice, None, Ellipsis]):
            Maximum `n` number of arguments to slice a tensor of rank `n` .
            One object in slices can be one of:

            1.  :py:obj:`int`: Slice this index only along the first dimension. Negative index is supported.
            2.  :py:obj:`list(int)`: Slice these indices along the first dimension. Negative indices are supported.
            3.  :py:obj:`slice`: Slice the generated indices from the
                `slice <https://docs.python.org/3.7/library/functions.html?highlight=slice#slice>`_ object along the
                first dimension. Similar to start:stop:step.
            4.  :py:obj:`None`: Slice the whole dimension. Similar to :py:obj:`[:]` in Python indexing.
            5.  :py:obj:`Ellipsis`: Slice the whole dimension, same result with `None` .

    Raises:
        TypeError: If `slices` is not of type int, list[int], :py:obj:`slice` , :py:obj:`None` or :py:obj:`Ellipsis` .

    Supported Platforms:
        ``CPU``

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

    @deprecated_c_transforms()
    @check_slice_op
    def __init__(self, *slices):
        slice_input_ = list(slices)
        slice_input_ = [_SliceOption(slice_dim) for slice_dim in slice_input_]
        self.slice_input_ = slice_input_

    def parse(self):
        return cde.SliceOperation(self.slice_input_)


class Relational(IntEnum):
    """
    Relationship operator.

    Possible enumeration values are: Relational.EQ, Relational.NE, Relational.GT, Relational.GE, Relational.LT,
    Relational.LE.

    - Relational.EQ: refers to Equality.
    - Relational.NE: refers not equal, or Inequality.
    - Relational.GT: refers to Greater than.
    - Relational.GE: refers to Greater than or equal to.
    - Relational.LT: refers to Less than.
    - Relational.LE: refers to Less than or equal to.
    """
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


class Mask(TensorOperation):
    r"""
    Mask content of the input tensor with the given predicate.
    Any element of the tensor that matches the predicate will be evaluated to True, otherwise False.

    Args:
        operator (Relational): relational operators, it can be any of [Relational.EQ, Relational.NE, Relational.LT,
            Relational.GT, Relational.LE, Relational.GE], take Relational.EQ as example, EQ refers to equal.
        constant (Union[str, int, float, bool]): Constant to be compared to.
        dtype (mindspore.dtype, optional): Type of the generated mask. Default: mstype.bool_.

    Raises:
        TypeError: `operator` is not of type Relational.
        TypeError: `constant` is not of type string int, float or bool.
        TypeError: `dtype` is not of type mindspore.dtype.

    Supported Platforms:
        ``CPU``

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

    @deprecated_c_transforms()
    @check_mask_op
    def __init__(self, operator, constant, dtype=mstype.bool_):
        self.operator = operator
        self.dtype = mstype_to_detype(dtype)
        self.constant = cde.Tensor(np.array(constant))

    def parse(self):
        return cde.MaskOperation(DE_C_RELATIONAL.get(self.operator), self.constant, self.dtype)


class PadEnd(TensorOperation):
    """
    Pad input tensor according to pad_shape, input tensor needs to have same rank.

    Args:
        pad_shape (list(int)): List of integers representing the shape needed. Dimensions that set to `None` will
            not be padded (i.e., original dim will be used). Shorter dimensions will truncate the values.
        pad_value (Union[str, bytes, int, float, bool], optional): Value used to pad. Default to 0 or empty
            string in case of tensors of strings.

    Raises:
        TypeError: If `pad_shape` is not of type list.
        TypeError: If `pad_value` is not of type str, float, bool, int or bytes.
        TypeError: If elements of `pad_shape` is not of type int.
        ValueError: If elements of `pad_shape` is not of positive.

    Supported Platforms:
        ``CPU``

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

    @deprecated_c_transforms()
    @check_pad_end
    def __init__(self, pad_shape, pad_value=None):
        self.pad_shape = cde.TensorShape(pad_shape)
        self.pad_value = cde.Tensor(np.array(pad_value)) if pad_value is not None else pad_value

    def parse(self):
        return cde.PadEndOperation(self.pad_shape, self.pad_value)


class Concatenate(TensorOperation):
    """
    Tensor operation that concatenates all columns into a single tensor.

    Args:
        axis (int, optional): Concatenate the tensors along given axis. Default: 0.
        prepend (numpy.array, optional): NumPy array to be prepended to the already concatenated tensors.
            Default: ``None``.
        append (numpy.array, optional): NumPy array to be appended to the already concatenated tensors.
            Default: ``None``.

    Raises:
        TypeError: If `axis` is not of type int.
        TypeError: If `prepend` is not of type numpy.ndarray.
        TypeError: If `append` is not of type numpy.ndarray.

    Supported Platforms:
        ``CPU``

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

    @deprecated_c_transforms()
    @check_concat_type
    def __init__(self, axis=0, prepend=None, append=None):
        self.axis = axis
        self.prepend = cde.Tensor(np.array(prepend)) if prepend is not None else prepend
        self.append = cde.Tensor(np.array(append)) if append is not None else append

    def parse(self):
        return cde.ConcatenateOperation(self.axis, self.prepend, self.append)


class Duplicate(TensorOperation):
    """
    Duplicate the input tensor to output, only support transform one column each time.

    Raises:
        RuntimeError: If given tensor has two columns.

    Supported Platforms:
        ``CPU``

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
        ...                                                 output_columns=["x", "y"])
        >>> # Data after
        >>> # |  x      |  y      |
        >>> # +---------+---------+
        >>> # | [1,2,3] | [1,2,3] |
        >>> # +---------+---------+
    """

    @deprecated_c_transforms()
    def __init__(self):
        super().__init__()

    def parse(self):
        return cde.DuplicateOperation()


class Unique(TensorOperation):
    """
    Perform the unique operation on the input tensor, only support transform one column each time.

    Return 3 tensor: unique output tensor, index tensor, count tensor.

    - Output tensor contains all the unique elements of the input tensor
      in the same order that they occur in the input tensor.
    - Index tensor that contains the index of each element of the input tensor in the unique output tensor.
    - Count tensor that contains the count of each element of the output tensor in the input tensor.

    Note:
        Call batch op before calling this function.

    Raises:
        RuntimeError: If given Tensor has two columns.

    Supported Platforms:
        ``CPU``

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
        ...                       output_columns=["x", "y", "z"])
        >>> # Data after
        >>> # |  x      |  y              |z        |
        >>> # +---------+-----------------+---------+
        >>> # | [0,1,2,3] | [0,1,2,1,2,3] | [1,2,2,1]
        >>> # +---------+-----------------+---------+
    """

    @deprecated_c_transforms()
    def __init__(self):
        super().__init__()

    def parse(self):
        return cde.UniqueOperation()


class Compose(TensorOperation):
    """
    Compose a list of transforms into a single transform.

    Args:
        transforms (list): List of transformations to be applied.

    Raises:
        TypeError: If `transforms` is not of type list.
        ValueError: If `transforms` is empty.
        TypeError: If elements of `transforms` are neither Python callable objects nor data
            processing operations in c_transforms.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> compose = c_transforms.Compose([c_vision.Decode(), c_vision.RandomCrop(512)])
        >>> image_folder_dataset = image_folder_dataset.map(operations=compose)
    """

    @deprecated_c_transforms()
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


class RandomApply(TensorOperation):
    """
    Randomly perform a series of transforms with a given probability.

    Args:
        transforms (list): List of transformations to be applied.
        prob (float, optional): The probability to apply the transformation list. Default: 0.5.

    Raises:
        TypeError: If `transforms` is not of type list.
        ValueError: If `transforms` is empty.
        TypeError: If elements of `transforms` are neither Python callable objects nor data
            processing operations in c_transforms.
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range [0.0, 1.0].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> rand_apply = c_transforms.RandomApply([c_vision.RandomCrop(512)])
        >>> image_folder_dataset = image_folder_dataset.map(operations=rand_apply)
    """

    @deprecated_c_transforms()
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


class RandomChoice(TensorOperation):
    """
    Randomly select one transform from a list of transforms to perform operation.

    Args:
        transforms (list): List of transformations to be chosen from to apply.

    Raises:
        TypeError: If `transforms` is not of type list.
        ValueError: If `transforms` is empty.
        TypeError: If elements of `transforms` are neither Python callable objects nor data
            processing operations in c_transforms.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> rand_choice = c_transforms.RandomChoice([c_vision.CenterCrop(50), c_vision.RandomCrop(512)])
        >>> image_folder_dataset = image_folder_dataset.map(operations=rand_choice)
    """

    @deprecated_c_transforms()
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


class Plugin(TensorOperation):
    """
    Plugin support for MindData. Use this class to dynamically load a .so file (shared library) and execute its symbols.

    Args:
        lib_path (str): Path to .so file which is compiled to support MindData plugin.
        func_name (str): Name of the function to load from the .so file.
        user_args (str, optional): Serialized args to pass to the plugin. Only needed if "func_name" requires one.

    Raises:
        TypeError: If `lib_path` is not of type string.
        TypeError: If `func_name` is not of type string.
        TypeError: If `user_args` is not of type string.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> plugin = c_transforms.Plugin("pluginlib.so", "PluginDecode")
        >>> image_folder_dataset = image_folder_dataset.map(operations=plugin)
    """

    @deprecated_c_transforms()
    @check_plugin
    def __init__(self, lib_path, func_name, user_args=None):
        self.lib_path = lib_path
        self.func_name = func_name
        self.user_args = str() if (user_args is None) else user_args

    def parse(self):
        return cde.PluginOperation(self.lib_path, self.func_name, self.user_args)
