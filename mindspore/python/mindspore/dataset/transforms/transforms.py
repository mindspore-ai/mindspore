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
The module transforms provides common operations, including Plugin and TypeCast.
"""
import json
from abc import ABC
import os
import threading

import sys
import numpy as np

import mindspore._c_dataengine as cde
from mindspore._c_expression import typing
from . import py_transforms_util as util
from .py_transforms_util import Implementation, FuncWrapper
from .validators import (
    check_slice_option,
    check_compose_call,
    check_random_transform_ops,
    check_plugin,
    check_type_cast,
)
from ..core.datatypes import mstype_to_detype, nptype_to_detype
from ..vision.py_transforms_util import is_pil


# hold all the executor objects when in training procedure
# key : pid_tid which distinguishes multiple executors by process_id + thread_id
# value : executor object which lifecycle will always exist during training
EXECUTORS_LIST = {}


# the follow case process / thread exit need call the function
# 1. user defined dataset with process mode
# 2. user defined dataset with thread mode
# 3. user defined transform in map op with process mode
# 4. user defined transform in map op with thread mode
# 5. batch op with per_batch_map operation in process mode
# 6. batch op with per_batch_map operation in thread mode
def clean_unused_executors():
    """
    clean the unused executor object in UDF or map with PyFunc process / thread mode
    """
    key = str(os.getpid()) + "_" + str(threading.current_thread().ident)
    if key in EXECUTORS_LIST:
        EXECUTORS_LIST.pop(key)


class TensorOperation:
    """
    Base class Tensor Ops
    """

    def __init__(self):
        super().__init__()
        self.implementation = None
        self.device_target = "CPU"

    def __call__(self, *input_tensor_list):
        """
        Call method.
        """
        # Check PIL Image with device_target
        if (len(input_tensor_list) == 1 and is_pil(input_tensor_list[0])) and self.device_target == "Ascend":
            raise TypeError("The input PIL Image cannot be executed on Ascend, "
                            "you can convert the input to the numpy ndarray type.")

        # Check if Python implementation of op, or PIL input
        if (self.implementation == Implementation.PY) or \
                (len(input_tensor_list) == 1 and is_pil(input_tensor_list[0]) and getattr(self, '_execute_py', None)):
            return self._execute_py(*input_tensor_list)

        tensor_row = []
        for tensor in input_tensor_list:
            try:
                tensor_row.append(cde.Tensor(np.asarray(tensor)))
            except (RuntimeError, TypeError) as exc:
                raise TypeError("Invalid user input. Got {}: {}, cannot be converted into tensor." \
                                .format(type(tensor), tensor)) from exc

        # get or create the executor from EXECUTORS_LIST
        executor = None
        key = str(os.getpid()) + "_" + str(threading.current_thread().ident)
        try:
            if key in EXECUTORS_LIST:
                # get the executor by process id and thread id
                executor = EXECUTORS_LIST[key]
                # remove the old transform which in executor and update the new transform
                executor.UpdateOperation(self.parse())
            else:
                # create a new executor by process id and thread_id
                executor = cde.Execute(self.parse())
                # add the executor the global EXECUTORS_LIST
                EXECUTORS_LIST[key] = executor

            output_tensor_list = executor(tensor_row)
        except RuntimeError as exc:
            if "Create stream failed" in str(exc):
                raise RuntimeError("Cannot reset NPU device in forked subprocess.\n    "
                                   "Note: the following several scenarios are not supported yet.\n"
                                   "    1. GeneratorDataset with num_parallel_workers>1 and "
                                   "python_multiprocessing=True.\n    2. Independent dataset mode (export "
                                   "MS_INDEPENDENT_DATASET=True):\n        1) Use the eager mode of dvpp "
                                   "in the main process, and then start the dataset independent process. "
                                   "GeneratorDataset / map / batch performs dvpp operations in thread mode.\n"
                                   "        2) Initialize the device in the main process, and then start the "
                                   "dataset independent process. GeneratorDataset / map / batch executes the "
                                   "dvpp operation in thread mode.\n    "
                                   "Suggestion: except for the scenes above to use NPU with multiprocessing, "
                                   "you can set ds.config.set_multiprocessing_start_method('spawn') in your "
                                   "script and rerun.") from exc
            raise exc from exc
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
            valid_module = ['mindspore.dataset.vision', 'mindspore.dataset.transforms']
            for json_op in json_obj["transforms"]:
                py_module = sys.modules.get(json_op.get("python_module"))
                if py_module is None or py_module.__package__ not in valid_module:
                    raise RuntimeError('Invalid json content, try to serialzie dataset again.')
                transforms.append(getattr(py_module, json_op["tensor_op_name"]).from_json(
                    json.dumps(json_op["tensor_op_params"])))
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
        super().__init__()
        self.transforms = []
        trans_with_imple = []
        for op in transforms:
            if callable(op) and not hasattr(op, "implementation"):
                op = util.FuncWrapper(op)
            if hasattr(op, "implementation"):
                if op.implementation is not None:
                    trans_with_imple.append(op)
            self.transforms.append(op)

        if all(t.implementation == Implementation.PY for t in self.transforms):
            self.implementation = Implementation.PY
        elif all(t.implementation is not None for t in self.transforms):
            self.implementation = Implementation.C
        elif not trans_with_imple:
            self.implementation = None
        elif all(t.implementation == Implementation.PY for t in trans_with_imple):
            self.implementation = Implementation.PY
        elif all(t.implementation == Implementation.C for t in trans_with_imple):
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


class _Compose(CompoundOperation):
    """
    _Compose a list of transforms into a single transform.

    .. Note::
        _Compose takes a list of transformations in `mindspore.dataset.transforms` / `mindspore.dataset.vision`
        and user-defined Python callable objects to combine as a single data augmentation.
        For user-defined Python callable objects, the return value is required to be of type numpy.ndarray.

    Args:
        transforms (list): List of transformations to be applied.

    Raises:
        TypeError: If `transforms` is not of type list.
        ValueError: If `transforms` is empty.
        TypeError: If elements of `transforms` are neither Python callable objects nor data
            processing operations in the transforms or vision modules.

    Supported Platforms:
        ``CPU``

    """

    @check_random_transform_ops
    def __init__(self, transforms):
        super().__init__(transforms)
        self.transforms = _Compose.decompose(self.transforms)
        if all(hasattr(transform, "random") and not transform.random for transform in self.transforms):
            self.random = False

    # pylint: disable=missing-docstring
    @staticmethod
    def decompose(operations):
        # Remove all compose operation from the given list of operations.
        #
        # Args:
        #    operations (list): list of transforms.
        #
        # Returns:
        #    list of operations without compose operations.
        new_operations = []
        for op in operations:
            if isinstance(op, _Compose):
                new_operations.extend(_Compose.decompose(op.transforms))
            else:
                new_operations.append(op)
        return new_operations

    # pylint: disable=missing-docstring
    @staticmethod
    def reduce(operations):
        # Wraps adjacent Python operations in a _Compose to allow mixing of Python and C++ operations.
        #
        # Args:
        #    operations (list): list of tensor operations.
        #
        # Returns:
        #    list, the reduced list of operations.
        new_ops, start_ind, end_ind = [], 0, 0
        for i, op in enumerate(operations):
            if op.implementation == Implementation.C and not isinstance(op, FuncWrapper):
                # reset counts
                if start_ind != end_ind:
                    if end_ind == start_ind + 1:
                        composed_op = operations[start_ind]
                    else:
                        composed_op = _Compose(operations[start_ind:end_ind])
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
                composed_op = _Compose(operations[start_ind:end_ind])
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

    def __call__(self, *args):
        '''
        If PY op exists in self.transforms, should use _execute_py to keep the output types unchanged.
        '''
        if any(t.implementation == Implementation.PY for t in self.transforms):
            self.implementation = Implementation.PY
        return super().__call__(*args)

    def release_resource(self):
        # release the executor which is used by current thread/process when
        # use transform in eager mode in map op
        # this will be call in MapOp::WorkerEntry
        clean_unused_executors()


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
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.transforms as transforms
        >>>
        >>> plugin = transforms.Plugin("pluginlib.so", "PluginDecode")
        >>> image_folder_dataset = ds.ImageFolderDataset("/path/to/image_folder_dataset_directory")
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


class TypeCast(TensorOperation):
    """
    Tensor operation to cast to a given MindSpore data type or NumPy data type.

    Note:
        This operation is executed on the CPU by default, but it can also be executed
        on the GPU or Ascend via heterogeneous acceleration.

    Args:
        data_type (Union[mindspore.dtype, numpy.dtype]): mindspore.dtype or numpy.dtype (e.g. `numpy.float32`)
            to be cast to.

    Raises:
        TypeError: If `data_type` is not of MindSpore data type bool, int, float, str or type `numpy.dtype`.

    Supported Platforms:
        ``CPU`` ``GPU`` ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.transforms as transforms
        >>> from mindspore import dtype as mstype
        >>>
        >>> # Use the transform in dataset pipeline mode
        >>> # Generate 1d int numpy array from 0 - 63
        >>> def generator_1d():
        ...     for i in range(64):
        ...         yield (np.array([i]),)
        >>>
        >>> generator_dataset = ds.GeneratorDataset(generator_1d, column_names='col')
        >>> type_cast_op = transforms.TypeCast(mstype.int32)
        >>> generator_dataset = generator_dataset.map(operations=type_cast_op)
        >>> for item in generator_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        ...     print(item["col"].shape, item["col"].dtype)
        ...     break
        (1,) int32
        >>>
        >>> # Use the transform in eager mode
        >>> data = np.array([2.71606445312564e-03, 6.3476562564e-03]).astype(np.float64)
        >>> output = transforms.TypeCast(np.float16)(data)
        >>> print(output.shape, output.dtype)
        (2,) float16
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
