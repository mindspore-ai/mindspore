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
The module transforms provides common operations, including Compose, OneHot and TypeCast.
"""
import json
from abc import ABC

import sys
from enum import IntEnum
import numpy as np

import mindspore._c_dataengine as cde
from mindspore._c_expression import typing
from mindspore.common import dtype as mstype
import mindspore.dataset.transforms.c_transforms as c_transforms
import mindspore.dataset.transforms.py_transforms as py_transforms
import mindspore.dataset.vision.c_transforms as c_vision
from . import py_transforms_util as util
from .py_transforms_util import Implementation, FuncWrapper
from .validators import check_fill_value, check_slice_option, check_slice_op, check_one_hot_op, check_compose_call, \
    check_mask_op_new, check_pad_end, check_concat_type, check_random_transform_ops, check_plugin, check_type_cast
from ..core.datatypes import mstype_to_detype, nptype_to_detype
from ..vision.py_transforms_util import is_pil


class TensorOperation:
    """
    Base class Tensor Ops
    """

    def __init__(self):
        super().__init__()
        self.implementation = None
        self.callable_op_ = None

    def __call__(self, *input_tensor_list):
        """
        Call method.
        """
        # Check if Python implementation of op, or PIL input
        if (self.implementation == Implementation.PY) or \
                (len(input_tensor_list) == 1 and is_pil(input_tensor_list[0]) and getattr(self, '_execute_py', None)):
            return self._execute_py(*input_tensor_list)

        tensor_row = []
        for tensor in input_tensor_list:
            try:
                tensor_row.append(cde.Tensor(np.asarray(tensor)))
            except (RuntimeError, TypeError):
                raise TypeError("Invalid user input. Got {}: {}, cannot be converted into tensor." \
                                .format(type(tensor), tensor))
        if not hasattr(self, 'callable_op_') or self.callable_op_ is None:
            self.callable_op_ = cde.Execute(self.parse())
        output_tensor_list = self.callable_op_(tensor_row)
        output_numpy_list = [x.as_array() for x in output_tensor_list]
        return output_numpy_list[0] if len(output_numpy_list) == 1 else tuple(output_numpy_list)

    @staticmethod
    def parse():
        """parse function - not yet implemented"""
        raise NotImplementedError("TensorOperation has to implement parse() method.")


class PyTensorOperation:
    """
    Base Python Tensor Operations class
    """

    def __init__(self):
        self.transforms = []
        self.output_type = None

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): Image to be augmented.

        Returns:
            PIL Image, augmented image.
        """
        return self._execute_py(img)

    @classmethod
    def from_json(cls, json_string):
        """
        Base from_json for Python tensor operations class
        """
        json_obj = json.loads(json_string)
        new_op = cls.__new__(cls)
        new_op.__dict__ = json_obj
        if "transforms" in json_obj.keys():
            # operations which have transforms as input, need to call _from_json() for each transform to deseriallize
            transforms = []
            for json_op in json_obj["transforms"]:
                transforms.append(getattr(
                    sys.modules.get(json_op.get("python_module")),
                    json_op["tensor_op_name"]).from_json(json.dumps(json_op["tensor_op_params"])))
            new_op.transforms = transforms
        if "output_type" in json_obj.keys():
            output_type = np.dtype(json_obj["output_type"])
            new_op.output_type = output_type
        return new_op

    def to_json(self):
        """
        Base to_json for Python tensor operations class
        """
        json_obj = {}
        json_trans = {}
        if "transforms" in self.__dict__.keys():
            # operations which have transforms as input, need to call _to_json() for each transform to serialize
            json_list = []
            for transform in self.transforms:
                json_list.append(json.loads(transform.to_json()))
            json_trans["transforms"] = json_list
            self.__dict__.pop("transforms")
        if "output_type" in self.__dict__.keys():
            json_trans["output_type"] = np.dtype(
                self.__dict__["output_type"]).name
            self.__dict__.pop("output_type")
        json_obj["tensor_op_params"] = self.__dict__
        # append transforms to the tensor_op_params of the operation
        json_obj.get("tensor_op_params").update(json_trans)
        json_obj["tensor_op_name"] = self.__class__.__name__
        json_obj["python_module"] = self.__class__.__module__
        return json.dumps(json_obj)


class CompoundOperation(TensorOperation, PyTensorOperation, ABC):
    """
    Compound Tensor Operations class
    """

    def __init__(self, transforms):
        super(CompoundOperation, self).__init__()
        self.transforms = []
        trans_with_imple = []
        for op in transforms:
            if callable(op) and not hasattr(op, "implementation") and \
                    not isinstance(op, c_transforms.TensorOperation) and \
                    not isinstance(op, py_transforms.PyTensorOperation) and \
                    not isinstance(op, c_vision.ImageTensorOperation):
                op = util.FuncWrapper(op)
            if hasattr(op, "implementation"):
                if op.implementation is not None:
                    trans_with_imple.append(op)
            else:
                raise RuntimeError("Mixing old legacy c/py_transforms and new unified transforms is not allowed.")
            self.transforms.append(op)

        if all([t.implementation == Implementation.PY for t in self.transforms]):
            self.implementation = Implementation.PY
        elif all([t.implementation is not None for t in self.transforms]):
            self.implementation = Implementation.C
        elif not trans_with_imple:
            self.implementation = None
        elif all([t.implementation == Implementation.PY for t in trans_with_imple]):
            self.implementation = Implementation.PY
        elif all([t.implementation == Implementation.C for t in trans_with_imple]):
            self.implementation = Implementation.C

    @staticmethod
    def parse():
        """parse function - not yet implemented"""
        raise NotImplementedError("CompoundOperation has to implement parse() method.")

    def parse_transforms(self):
        operations = []
        for op in self.transforms:
            if op and getattr(op, 'parse', None):
                operations.append(op.parse())
            else:
                operations.append(op)
        return operations


def not_random(function):
    """
    Specify the function as "not random", i.e., it produces deterministic result.
    A Python function can only be cached after it is specified as "not random".
    """
    function.random = False
    return function


class Compose(CompoundOperation):
    """
    Compose a list of transforms into a single transform.

    .. Note::
        Compose takes a list of transformations in `mindspore.dataset.transforms` / `mindspore.dataset.vision`
        and user-defined Python callable objects to combine as single data augmentation.
        For user-defined Python callable objects, the return value is required to be type numpy.ndarray.

    Args:
        transforms (list): List of transformations to be applied.

    Raises:
        TypeError: If `transforms` is not of type list.
        ValueError: If `transforms` is empty.
        TypeError: If elements of `transforms` are neither Python callable objects nor data
            processing operations in transforms.py.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> compose = transforms.Compose([vision.Decode(), vision.RandomCrop(512)])
        >>> image_folder_dataset = image_folder_dataset.map(operations=compose)
        >>> image_folder_dataset_dir = "/path/to/image_folder_dataset_directory"
        >>>
        >>> # create a dataset that reads all files in dataset_dir with 8 threads
        >>> image_folder_dataset = ds.ImageFolderDataset(image_folder_dataset_dir, num_parallel_workers=8)
        >>> # create a list of transformations to be applied to the image data
        >>> transform = transforms.Compose([vision.Decode(to_pil=True),
        ...                                vision.RandomHorizontalFlip(0.5),
        ...                                vision.ToTensor(),
        ...                                vision.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262), is_hwc=False),
        ...                                vision.RandomErasing()])
        >>> # apply the transform to the dataset through dataset.map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transform, input_columns=["image"])
        >>>
        >>> # Compose is also be invoked implicitly, by just passing in a list of ops
        >>> # the above example then becomes:
        >>> transforms_list = [vision.Decode(to_pil=True),
        ...                    vision.RandomHorizontalFlip(0.5),
        ...                    vision.ToTensor(),
        ...                    vision.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262), is_hwc=False),
        ...                    vision.RandomErasing()]
        >>>
        >>> # apply the transform to the dataset through dataset.map()
        >>> image_folder_dataset_1 = image_folder_dataset_1.map(operations=transforms_list, input_columns=["image"])
        >>>
        >>> # Certain C++ and Python ops can be combined, but not all of them
        >>> # An example of combined operations
        >>> arr = [0, 1]
        >>> dataset = ds.NumpySlicesDataset(arr, column_names=["cols"], shuffle=False)
        >>> transformed_list = [transforms.OneHot(2),
        ...                     transforms.Mask(transforms.Relational.EQ, 1)]
        >>> dataset = dataset.map(operations=transformed_list, input_columns=["cols"])
        >>>
        >>> # Here is an example of mixing vision ops
        >>> import numpy as np
        >>> op_list=[vision.Decode(),
        ...          vision.Resize((224, 244)),
        ...          vision.ToPIL(),
        ...          np.array, # need to convert PIL image to a NumPy array to pass it to C++ operation
        ...          vision.Resize((24, 24))]
        >>> image_folder_dataset = image_folder_dataset.map(operations=op_list,  input_columns=["image"])
    """

    @check_random_transform_ops
    def __init__(self, transforms):
        super().__init__(transforms)
        self.transforms = Compose.decompose(self.transforms)
        if all(hasattr(transform, "random") and not transform.random for transform in self.transforms):
            self.random = False

    @staticmethod
    def decompose(operations):
        """
        Remove all compose operation from the given list of operations.

        Args:
            operations (list): list of transforms.

        Returns:
            list of operations without compose operations.
        """
        new_operations = []
        for op in operations:
            if isinstance(op, Compose):
                new_operations.extend(Compose.decompose(op.transforms))
            else:
                new_operations.append(op)
        return new_operations

    @staticmethod
    def reduce(operations):
        """
        Wraps adjacent Python operations in a Compose to allow mixing of Python and C++ operations.

        Args:
            operations (list): list of tensor operations.

        Returns:
            list, the reduced list of operations.
        """
        new_ops, start_ind, end_ind = [], 0, 0
        for i, op in enumerate(operations):
            if op.implementation == Implementation.C and not isinstance(op, FuncWrapper):
                # reset counts
                if start_ind != end_ind:
                    if end_ind == start_ind + 1:
                        composed_op = operations[start_ind]
                    else:
                        composed_op = Compose(operations[start_ind:end_ind])
                        composed_op.implementation = Implementation.PY
                    new_ops.append(composed_op)
                new_ops.append(op)
                start_ind, end_ind = i + 1, i + 1
            else:
                end_ind += 1
        # do additional check in case the last operation is a Python operation
        if start_ind != end_ind:
            if end_ind == start_ind + 1:
                composed_op = operations[start_ind]
            else:
                composed_op = Compose(operations[start_ind:end_ind])
                composed_op.implementation = Implementation.PY
            new_ops.append(composed_op)
        return new_ops

    def parse(self):
        operations = self.parse_transforms()
        return cde.ComposeOperation(operations)

    @check_compose_call
    def _execute_py(self, *args):
        """
        Execute method.

        Returns:
            lambda function, Lambda function that takes in an args to apply transformations on.
        """
        return util.compose(self.transforms, *args)


class Concatenate(TensorOperation):
    """
    Tensor operation that concatenates all columns into a single tensor, only 1D tenspr is supported.

    Args:
        axis (int, optional): Concatenate the tensors along given axis. Default: 0.
        prepend (numpy.ndarray, optional): NumPy array to be prepended to the already concatenated tensors.
            Default: None.
        append (numpy.ndarray, optional): NumPy array to be appended to the already concatenated tensors. Default: None.

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
        >>> concatenate_op = transforms.Concatenate(0, prepend_tensor, append_tensor)
        >>> data = [["This","is","a","string"]]
        >>> dataset = ds.NumpySlicesDataset(data)
        >>> dataset = dataset.map(operations=concatenate_op)
    """

    @check_concat_type
    def __init__(self, axis=0, prepend=None, append=None):
        super().__init__()
        self.axis = axis
        self.prepend = cde.Tensor(np.array(prepend)) if prepend is not None else prepend
        self.append = cde.Tensor(np.array(append)) if append is not None else append
        self.implementation = Implementation.C

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
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms.Duplicate(),
        ...                                                 input_columns=["x"],
        ...                                                 output_columns=["x", "y"])
        >>> # Data after
        >>> # |  x      |  y      |
        >>> # +---------+---------+
        >>> # | [1,2,3] | [1,2,3] |
        >>> # +---------+---------+
    """

    def __init__(self):
        super().__init__()
        self.implementation = Implementation.C

    def parse(self):
        return cde.DuplicateOperation()


class Fill(TensorOperation):
    """
    Tensor operation to fill all elements in the tensor with the specified value.
    The output tensor will have the same shape and type as the input tensor.

    Args:
        fill_value (Union[str, bytes, int, float, bool]): scalar value
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
        >>> fill_op = transforms.Fill(3)
        >>> generator_dataset = generator_dataset.map(operations=fill_op)
        >>> # [[3], [3], [3], [3], [3]]
    """

    @check_fill_value
    def __init__(self, fill_value):
        super().__init__()
        self.fill_value = cde.Tensor(np.array(fill_value))
        self.implementation = Implementation.C

    def parse(self):
        return cde.FillOperation(self.fill_value)


class Mask(TensorOperation):
    r"""
    Mask content of the input tensor with the given predicate.
    Any element of the tensor that matches the predicate will be evaluated to True, otherwise False.

    Args:
        operator (Relational): relational operators, it can be any of [Relational.EQ, Relational.NE, Relational.LT,
            Relational.GT, Relational.LE, Relational.GE], take Relational.EQ as example, EQ refers to equal.
        constant (Union[str, int, float, bool]): Constant to be compared to.
        dtype (mindspore.dtype, optional): Type of the generated mask. Default: mindspore.dtype.bool\_.

    Raises:
        TypeError: `operator` is not of type Relational.
        TypeError: `constant` is not of type string int, float or bool.
        TypeError: `dtype` is not of type mindspore.dtype.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms import Relational
        >>> # Data before
        >>> # |  col   |
        >>> # +---------+
        >>> # | [1,2,3] |
        >>> # +---------+
        >>> data = [[1, 2, 3]]
        >>> numpy_slices_dataset = ds.NumpySlicesDataset(data, ["col"])
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms.Mask(Relational.EQ, 2))
        >>> # Data after
        >>> # |       col         |
        >>> # +--------------------+
        >>> # | [False,True,False] |
        >>> # +--------------------+
    """

    @check_mask_op_new
    def __init__(self, operator, constant, dtype=mstype.bool_):
        super().__init__()
        self.operator = operator
        self.dtype = mstype_to_detype(dtype)
        self.constant = cde.Tensor(np.array(constant))
        self.implementation = Implementation.C

    def parse(self):
        return cde.MaskOperation(DE_C_RELATIONAL.get(self.operator), self.constant, self.dtype)


class OneHot(TensorOperation):
    """
    Tensor operation to apply one hot encoding.

    Args:
        num_classes (int): Number of classes of objects in dataset.
            It should be larger than the largest label number in the dataset.
        smoothing_rate (float, optional): Adjustable hyperparameter for label smoothing level.
            Default: 0.0, means no smoothing is applied.

    Raises:
        TypeError: `num_classes` is not of type int.
        TypeError: `smoothing_rate` is not of type float or int.
        ValueError: `smoothing_rate` is not in range [0.0, 1.0].
        RuntimeError: Input tensor is not of type int.
        RuntimeError: Input tensor is not a 1-D tensor.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> # Assume that dataset has 10 classes, thus the label ranges from 0 to 9
        >>> onehot_op = transforms.OneHot(num_classes=10)
        >>> mnist_dataset = mnist_dataset.map(operations=onehot_op, input_columns=["label"])
    """

    @check_one_hot_op
    def __init__(self, num_classes, smoothing_rate=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.random = False
        self.smoothing_rate = smoothing_rate

    def parse(self):
        return cde.OneHotOperation(self.num_classes, self.smoothing_rate)


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
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms.PadEnd(pad_shape=[4],
        ...                                                                                pad_value=10))
        >>> # Data after
        >>> # |    col     |
        >>> # +------------+
        >>> # | [1,2,3,10] |
        >>> # +------------|
    """

    @check_pad_end
    def __init__(self, pad_shape, pad_value=None):
        super().__init__()
        self.pad_shape = cde.TensorShape(pad_shape)
        self.pad_value = cde.Tensor(np.array(pad_value)) if pad_value is not None else pad_value
        self.implementation = Implementation.C

    def parse(self):
        return cde.PadEndOperation(self.pad_shape, self.pad_value)


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
        >>> plugin = transforms.Plugin("pluginlib.so", "PluginDecode")
        >>> image_folder_dataset = image_folder_dataset.map(operations=plugin)
    """

    @check_plugin
    def __init__(self, lib_path, func_name, user_args=None):
        super().__init__()
        self.lib_path = lib_path
        self.func_name = func_name
        self.user_args = str() if (user_args is None) else user_args
        self.implementation = Implementation.C

    def parse(self):
        return cde.PluginOperation(self.lib_path, self.func_name, self.user_args)


class RandomApply(CompoundOperation):
    """
    Randomly perform a series of transforms with a given probability.

    Args:
        transforms (list): List of transformations to be applied.
        prob (float, optional): The probability to apply the transformation list. Default: 0.5.

    Raises:
        TypeError: If `transforms` is not of type list.
        ValueError: If `transforms` is empty.
        TypeError: If elements of `transforms` are neither Python callable objects nor data
            processing operations in transforms.py.
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range [0.0, 1.0].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms import Compose
        >>> transforms_list = [vision.RandomHorizontalFlip(0.5),
        ...                    vision.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        ...                    vision.RandomErasing()]
        >>> composed_transform = Compose([vision.Decode(to_pil=True),
        ...                               transforms.RandomApply(transforms_list, prob=0.6),
        ...                               vision.ToTensor()])
        >>> image_folder_dataset = image_folder_dataset.map(operations=composed_transform, input_columns=["image"])
    """

    @check_random_transform_ops
    def __init__(self, transforms, prob=0.5):
        super().__init__(transforms)
        self.prob = prob

    def parse(self):
        operations = self.parse_transforms()
        return cde.RandomApplyOperation(self.prob, operations)

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (PIL image): Image to be randomly applied a list transformations.

        Returns:
            img (PIL image), Transformed image.
        """
        return util.random_apply(img, self.transforms, self.prob)


class RandomChoice(CompoundOperation):
    """
    Randomly select one transform from a list of transforms to perform operation.

    Args:
        transforms (list): List of transformations to be chosen from to apply.

    Raises:
        TypeError: If `transforms` is not of type list.
        ValueError: If `transforms` is empty.
        TypeError: If elements of `transforms` are neither Python callable objects nor data
            processing operations in transforms.py.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms import Compose
        >>> transforms_list = [vision.RandomHorizontalFlip(0.5),
        ...                    vision.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        ...                    vision.RandomErasing()]
        >>> composed_transform = Compose([vision.Decode(),
        ...                               transforms.RandomChoice(transforms_list),
        ...                               vision.ToTensor()])
        >>> image_folder_dataset = image_folder_dataset.map(operations=composed_transform, input_columns=["image"])

    """

    @check_random_transform_ops
    def __init__(self, transforms):
        super().__init__(transforms)

    def parse(self):
        operations = self.parse_transforms()
        return cde.RandomChoiceOperation(operations)

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (PIL image): Image to be applied transformation.


        Returns:
            img (PIL image), Transformed image.
        """
        return util.random_choice(img, self.transforms)


class RandomOrder(PyTensorOperation):
    """
    Perform a series of transforms to the input image in a random order.

    Args:
        transforms (list): List of the transformations to apply.

    Raises:
        TypeError: If `transforms` is not of type list.
        TypeError: If elements of `transforms` are neither Python callable objects nor data
            processing operations in mindspore.dataset.transforms.transforms.
        ValueError: If `transforms` is empty.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.dataset.transforms import Compose
        >>> transforms_list = [vision.RandomHorizontalFlip(0.5),
        ...                    vision.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        ...                    vision.RandomErasing()]
        >>> composed_transform = Compose([vision.Decode(to_pil=False),
        ...                               transforms.RandomOrder(transforms_list),
        ...                               vision.ToTensor()])
        >>> image_folder_dataset = image_folder_dataset.map(operations=composed_transform, input_columns=["image"])
    """

    @check_random_transform_ops
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms
        self.implementation = Implementation.PY

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (PIL image): Image to apply transformations in a random order.

        Returns:
            img (PIL image), Transformed image.
        """
        return util.random_order(img, self.transforms)


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
        >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms.Slice(slice(1,3)))
        >>> # Data after
        >>> # |   col   |
        >>> # +---------+
        >>> # |  [2,3]  |
        >>> # +---------|
    """

    @check_slice_op
    def __init__(self, *slices):
        super().__init__()
        slice_input_ = list(slices)
        slice_input_ = [_SliceOption(slice_dim) for slice_dim in slice_input_]
        self.slice_input_ = slice_input_
        self.implementation = Implementation.C

    def parse(self):
        return cde.SliceOperation(self.slice_input_)


class TypeCast(TensorOperation):
    """
    Tensor operation to cast to a given MindSpore data type or NumPy data type.

    Note:
        This operation supports running on Ascend or GPU platforms by Offload.

    Args:
        data_type (Union[mindspore.dtype, numpy.dtype]): mindspore.dtype or numpy.dtype (e.g. `numpy.float32`)
            to be cast to.

    Raises:
        TypeError: If `data_type` is not of MindSpore data type bool, int, float, string or type :class:`numpy.dtype` .

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
        >>> type_cast_op = transforms.TypeCast(mstype.int32)
        >>> dataset = dataset.map(operations=type_cast_op)
    """

    @check_type_cast
    def __init__(self, data_type):
        super().__init__()
        if isinstance(data_type, typing.Type):
            data_type = mstype_to_detype(data_type)
        else:
            data_type = nptype_to_detype(data_type)
        self.data_type = str(data_type)
        self.implementation = Implementation.C

    def parse(self):
        return cde.TypeCastOperation(self.data_type)


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
        >>> dataset = dataset.map(operations=transforms.Unique(),
        ...                       input_columns=["x"],
        ...                       output_columns=["x", "y", "z"])
        >>> # Data after
        >>> # |  x      |  y              |z        |
        >>> # +---------+-----------------+---------+
        >>> # | [0,1,2,3] | [0,1,2,1,2,3] | [1,2,2,1]
        >>> # +---------+-----------------+---------+
    """

    def __init__(self):
        super().__init__()
        self.implementation = Implementation.C

    def parse(self):
        return cde.UniqueOperation()
