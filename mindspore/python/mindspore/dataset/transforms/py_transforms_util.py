# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
Built-in py_transforms_utils functions.
"""
import json
import random
from enum import IntEnum
from types import FunctionType, MethodType
import numpy as np

from mindspore import log as logger
from ..core.py_util_helpers import is_numpy, ExceptionHandler
from .. import transforms as t


class Implementation(IntEnum):
    """
    Implementation types for operations

    - Implementation.C: the operation is implemented in C++
    - Implementation.PY: the operation is implemented in Python
    """
    C = 0
    PY = 1


def all_numpy(args):
    """ for multi-input lambdas"""
    if isinstance(args, tuple):
        for value in args:
            if not is_numpy(value):
                return False
        return True
    return is_numpy(args)


def compose(transforms, *args):
    """
    Compose a list of transforms and apply on the image.

    Args:
        img (numpy.ndarray): An image in NumPy ndarray.
        transforms (list): A list of transform Class objects to be composed.

    Returns:
        img (numpy.ndarray), An augmented image in NumPy ndarray.
    """
    for transform in transforms:
        try:
            args = transform(*args)
        except Exception:
            result = ExceptionHandler(where="in map(or batch) worker and execute Python function")
            result.reraise()
        args = (args,) if not isinstance(args, tuple) else args

    return args


def one_hot_encoding(label, num_classes, epsilon):
    """
    Apply label smoothing transformation to the input label, and make label be more smoothing and continuous.

    Args:
        label (numpy.ndarray): label to be applied label smoothing.
        num_classes (int): Num class of object in dataset, value should over 0.
        epsilon (float): The adjustable Hyper parameter. Default is 0.0.

    Returns:
        img (numpy.ndarray), label after being one hot encoded and done label smoothed.

    Examples:
        >>> # assume num_classes = 5
        >>> # 1) input np.array(3) output [0, 0, 0, 1, 0]
        >>> # 2) input np.array([4, 2, 0]) output [[0, 0, 0, 0, 1], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]]
        >>> # 3) input np.array([[4], [2], [0]]) output [[[0, 0, 0, 0, 1]], [[0, 0, 1, 0, 0][, [[1, 0, 0, 0, 0]]]
    """
    if isinstance(label, np.ndarray):  # the numpy should be () or (1, ) or shape: (n, 1)
        if label.dtype not in [np.int8, np.int16, np.int32, np.int64,
                               np.uint8, np.uint16, np.uint32, np.uint64]:
            raise ValueError('the input numpy type should be int, but the input is: ' + str(label.dtype))

        if label.ndim == 0:
            if label >= num_classes:
                raise ValueError('the num_classes is smaller than the category number.')

            one_hot_label = np.zeros((num_classes), dtype=int)
            one_hot_label[label] = 1
        else:
            label_flatten = label.flatten()
            for item in label_flatten:
                if item >= num_classes:
                    raise ValueError('the num_classes:' + str(num_classes) +
                                     ' is smaller than the category number:' + str(item))

            num_elements = label_flatten.size
            one_hot_label = np.zeros((num_elements, num_classes), dtype=int)
            for index in range(num_elements):
                one_hot_label[index][label_flatten[index]] = 1

            new_shape = []
            for dim in label.shape:
                new_shape.append(dim)
            new_shape.append(num_classes)
            one_hot_label = one_hot_label.reshape(new_shape)
    else:
        raise ValueError('the input is invalid, it should be numpy.ndarray.')

    return (1 - epsilon) * one_hot_label + epsilon / num_classes


def random_order(img, transforms):
    """
    Applies a list of transforms in a random order.

    Args:
        img: Image to be applied transformations in a random order.
        transforms (list): List of the transformations to be applied.

    Returns:
        img, Transformed image.
    """
    random.shuffle(transforms)
    for transform in transforms:
        img = transform(img)
    return img


def random_apply(img, transforms, prob):
    """
    Apply a list of transformation, randomly with a given probability.

    Args:
        img: Image to be randomly applied a list transformations.
        transforms (list): List of transformations to be applied.
        prob (float): The probability to apply the transformation list.

    Returns:
        img, Transformed image.
    """
    if prob < random.random():
        return img
    for transform in transforms:
        img = transform(img)
    return img


def random_choice(img, transforms):
    """
    Random selects one transform from a list of transforms and applies that on the image.

    Args:
        img: Image to be applied transformation.
        transforms (list): List of transformations to be chosen from to apply.

    Returns:
        img, Transformed image.
    """
    return random.choice(transforms)(img)


class FuncWrapper:
    """
    Wrap function with try except logic, mainly for wrapping Python function.

    Args:
        transform: Callable Python function.

    Returns:
        result, data after apply transformation.
    """

    def __init__(self, transform):
        if not callable(transform):
            raise ValueError("Input operations should be callable Python function, but got: " + str(transform))
        self.transform = transform
        self.implementation = Implementation.C
        try:
            if hasattr(self.transform, "random") and not self.transform.random:
                self.random = False
        except KeyError:
            self.random = True
        self.logged_list_mixed_type_warning = False  # Warning for list mixed type result is not logged yet

    def __call__(self, *args):
        try:
            result = self.transform(*args)
        except Exception:
            result = ExceptionHandler(where="in map(or batch) worker and execute Python function")
            result.reraise()

        # Check if result is list type, and mixed type warning for list type result has not been logged yet
        if isinstance(result, list) and not self.logged_list_mixed_type_warning:
            result0_type = type(result[0])
            if not all((type(t) is result0_type) for t in result):  # pylint: disable=unidiomatic-typecheck
                self.logged_list_mixed_type_warning = True
                warn_msg = "All elements in returned list are not of the same type in Python function." + \
                           " First element has type: " + str(result0_type)
                logger.warning(warn_msg)
        return result

    def to_json(self):
        """ Serialize to JSON format """
        # User-defined Python functions cannot be fully nor correctly serialized.
        # Log a warning, and produce minimal info for the Python UDF, so that serialization of the
        # dataset pipeline can continue.
        # Note that the serialized JSON output is not valid to be deserialized.
        udf_python_warning = "Serialization of user-defined Python functions is not supported. " \
                             "Any produced serialized JSON file output for this dataset pipeline is not valid " \
                             "to be deserialized."
        try:
            if isinstance(self.transform, (FunctionType, MethodType)):
                # common function type(include lambda, class method) / object method
                logger.warning(udf_python_warning)
                json_obj = {}
                json_obj["tensor_op_name"] = self.transform.__name__
                json_obj["python_module"] = self.__class__.__module__
                return json.dumps(json_obj)
            if callable(self.transform) and not isinstance(self.transform, (t.c_transforms.TensorOperation,
                                                                            t.py_transforms.PyTensorOperation,
                                                                            t.transforms.TensorOperation,
                                                                            t.transforms.PyTensorOperation,
                                                                            FuncWrapper)):
                # udf callable class
                logger.warning(udf_python_warning)
                json_obj = {}
                json_obj["tensor_op_name"] = type(self.transform).__name__
                json_obj["python_module"] = self.__class__.__module__
                return json.dumps(json_obj)
            # dataset operations
            return self.transform.to_json()
        except Exception as e:
            logger.warning("Skip user-defined Python method which cannot be serialized, reason is: " + str(e))
            json_obj = {}
            json_obj["tensor_op_name"] = "unknown"
            json_obj["python_module"] = "unknown"
            return json.dumps(json_obj)

    def release_resource(self):
        # release the executor which is used by current thread/process when
        # use transform in eager mode in map op or batch op
        # this will be call in MapOp::WorkerEntry and BatchOp::WorkerEntry
        t.transforms.clean_unused_executors()
