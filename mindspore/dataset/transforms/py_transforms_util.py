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
# ==============================================================================
"""
Built-in py_transforms_utils functions.
"""
import json
import random
from types import FunctionType
import numpy as np

from ..core.py_util_helpers import is_numpy, ExceptionHandler


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

    if all_numpy(args):
        return args
    raise TypeError('args should be NumPy ndarray. Got {}. Append ToTensor() to transforms.'.format(type(args)))


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
    Wrap function with try except logic, mainly for warping python function.

    Args:
        transform: Callable python function.

    Returns:
        result, data after apply transformation.
    """
    def __init__(self, transform):
        if not callable(transform):
            raise ValueError("Input operations should be callable python function, but got: " + str(transform))
        self.transform = transform
        try:
            if hasattr(self.transform, "random") and not self.transform.random:
                self.random = False
        except KeyError:
            self.random = True

    def __call__(self, *args):
        result = None
        try:
            result = self.transform(*args)
        except Exception:
            result = ExceptionHandler(where="in map(or batch) worker and execute python function")
            result.reraise()
        return result

    def to_json(self):
        if isinstance(self.transform, FunctionType):
            json_obj = {}
            json_obj["tensor_op_name"] = self.transform.__name__
            json_obj["python_module"] = self.__class__.__module__
            return json.dumps(json_obj)
        return self.transform.to_json()
