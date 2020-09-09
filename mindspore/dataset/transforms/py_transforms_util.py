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
import numpy as np
from ..core.py_util_helpers import is_numpy


def compose(img, transforms):
    """
    Compose a list of transforms and apply on the image.

    Args:
        img (numpy.ndarray): An image in Numpy ndarray.
        transforms (list): A list of transform Class objects to be composed.

    Returns:
        img (numpy.ndarray), An augmented image in Numpy ndarray.
    """
    if is_numpy(img):
        for transform in transforms:
            img = transform(img)
        if is_numpy(img):
            return img
        raise TypeError('img should be Numpy ndarray. Got {}. Append ToTensor() to transforms'.format(type(img)))
    raise TypeError('img should be Numpy ndarray. Got {}.'.format(type(img)))


def one_hot_encoding(label, num_classes, epsilon):
    """
    Apply label smoothing transformation to the input label, and make label be more smoothing and continuous.

    Args:
        label (numpy.ndarray): label to be applied label smoothing.
        num_classes (int): Num class of object in dataset, value should over 0.
        epsilon (float): The adjustable Hyper parameter. Default is 0.0.

    Returns:
        img (numpy.ndarray), label after being one hot encoded and done label smoothed.
    """
    if label > num_classes:
        raise ValueError('the num_classes is smaller than the category number.')

    num_elements = label.size
    one_hot_label = np.zeros((num_elements, num_classes), dtype=int)

    if isinstance(label, list) is False:
        label = [label]
    for index in range(num_elements):
        one_hot_label[index, label[index]] = 1

    return (1 - epsilon) * one_hot_label + epsilon / num_classes
