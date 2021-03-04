# Copyright 2019 Huawei Technologies Co., Ltd
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
The module transforms.py_transform is implemented based on Python. It provides common
operations including OneHotOp.
"""
from .validators import check_one_hot_op, check_compose_list, check_random_apply, check_transforms_list, \
    check_compose_call
from . import py_transforms_util as util


def not_random(function):
    function.random = False
    return function


class OneHotOp:
    """
    Apply one hot encoding transformation to the input label, make label be more smoothing and continuous.

    Args:
        num_classes (int): Number of classes of objects in dataset.
            It should be larger than the largest label number in the dataset.
        smoothing_rate (float, optional): Adjustable hyperparameter for label smoothing level.
            (Default=0.0 means no smoothing is applied.)

    Examples:
        >>> # Assume that dataset has 10 classes, thus the label ranges from 0 to 9
        >>> transforms_list = [py_transforms.OneHotOp(num_classes=10, smoothing_rate=0.1)]
        >>> transform = py_transforms.Compose(transforms_list)
        >>> mnist_dataset = mnist_dataset.map(input_columns=["label"], operations=transform)
    """

    @check_one_hot_op
    def __init__(self, num_classes, smoothing_rate=0.0):
        self.num_classes = num_classes
        self.smoothing_rate = smoothing_rate
        self.random = False

    def __call__(self, label):
        """
        Call method.

        Args:
            label (numpy.ndarray): label to be applied label smoothing.

        Returns:
            label (numpy.ndarray), label after being Smoothed.
        """
        return util.one_hot_encoding(label, self.num_classes, self.smoothing_rate)


class Compose:
    """
    Compose a list of transforms.

    .. Note::
        Compose takes a list of transformations either provided in py_transforms or from user-defined implementation;
        each can be an initialized transformation class or a lambda function, as long as the output from the last
        transformation is a single tensor of type numpy.ndarray. See below for an example of how to use Compose
        with py_transforms classes and check out FiveCrop or TenCrop for the use of them in conjunction with lambda
        functions.

    Args:
        transforms (list): List of transformations to be applied.

    Examples:
        >>> image_folder_dataset_dir = "/path/to/image_folder_dataset_directory"
        >>> # create a dataset that reads all files in dataset_dir with 8 threads
        >>> image_folder_dataset = ds.ImageFolderDataset(image_folder_dataset_dir, num_parallel_workers=8)
        >>> # create a list of transformations to be applied to the image data
        >>> transform = py_transforms.Compose([py_vision.Decode(),
        ...                                    py_vision.RandomHorizontalFlip(0.5),
        ...                                    py_vision.ToTensor(),
        ...                                    py_vision.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        ...                                    py_vision.RandomErasing()])
        >>> # apply the transform to the dataset through dataset.map function
        >>> image_folder_dataset = image_folder_dataset.map(operations=transform, input_columns=["image"])
        >>>
        >>> # Compose is also be invoked implicitly, by just passing in a list of ops
        >>> # the above example then becomes:
        >>> transforms_list = [py_vision.Decode(),
        ...                    py_vision.RandomHorizontalFlip(0.5),
        ...                    py_vision.ToTensor(),
        ...                    py_vision.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        ...                    py_vision.RandomErasing()]
        >>>
        >>> # apply the transform to the dataset through dataset.map()
        >>> image_folder_dataset_1 = image_folder_dataset_1.map(operations=transforms_list, input_columns=["image"])
        >>>
        >>> # Certain C++ and Python ops can be combined, but not all of them
        >>> # An example of combined operations
        >>> arr = [0, 1]
        >>> dataset = ds.NumpySlicesDataset(arr, column_names=["cols"], shuffle=False)
        >>> transformed_list = [py_transforms.OneHotOp(2), c_transforms.Mask(c_transforms.Relational.EQ, 1)]
        >>> dataset = dataset.map(operations=transformed_list, input_columns=["cols"])
        >>>
        >>> # Here is an example of mixing vision ops
        >>> import numpy as np
        >>> op_list=[c_vision.Decode(),
        ...          c_vision.Resize((224, 244)),
        ...          py_vision.ToPIL(),
        ...          np.array, # need to convert PIL image to a NumPy array to pass it to C++ operation
        ...          c_vision.Resize((24, 24))]
        >>> image_folder_dataset = image_folder_dataset.map(operations=op_list,  input_columns=["image"])
    """

    @check_compose_list
    def __init__(self, transforms):
        self.transforms = transforms
        if all(hasattr(transform, "random") and not transform.random for transform in self.transforms):
            self.random = False

    @check_compose_call
    def __call__(self, *args):
        """
        Call method.

        Returns:
            lambda function, Lambda function that takes in an args to apply transformations on.
        """
        return util.compose(self.transforms, *args)

    @staticmethod
    def reduce(operations):
        """
        Wraps adjacent Python operations in a Compose to allow mixing of Python and C++ operations
        Args:
            operations (list): list of tensor operations

        Returns:
            list, the reduced list of operations
        """
        #
        if len(operations) == 1:
            return operations

        new_ops, start_ind, end_ind = [], 0, 0
        for i, op in enumerate(operations):
            if str(op).find("c_transform") >= 0:
                # reset counts
                if start_ind != end_ind:
                    new_ops.append(Compose(operations[start_ind:end_ind]))
                new_ops.append(op)
                start_ind, end_ind = i + 1, i + 1
            else:
                end_ind += 1
        # do additional check in case the last operation is a Python operation
        if start_ind != end_ind:
            new_ops.append(Compose(operations[start_ind:end_ind]))
        return new_ops


class RandomApply:
    """
    Randomly perform a series of transforms with a given probability.

    Args:
        transforms (list): List of transformations to apply.
        prob (float, optional): The probability to apply the transformation list (default=0.5).

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = [py_vision.RandomHorizontalFlip(0.5),
        ...                    py_vision.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        ...                    py_vision.RandomErasing()]
        >>> transforms = Compose([py_vision.Decode(),
        ...                       py_transforms.RandomApply(transforms_list, prob=0.6),
        ...                       py_vision.ToTensor()])
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms, input_columns=["image"])
    """

    @check_random_apply
    def __init__(self, transforms, prob=0.5):
        self.prob = prob
        self.transforms = transforms

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be randomly applied a list transformations.

        Returns:
            img (PIL image), Transformed image.
        """
        return util.random_apply(img, self.transforms, self.prob)


class RandomChoice:
    """
    Randomly select one transform from a series of transforms and applies that on the image.

    Args:
         transforms (list): List of transformations to be chosen from to apply.

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose, RandomChoice
        >>> transforms_list = [py_vision.RandomHorizontalFlip(0.5),
        ...                    py_vision.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        ...                    py_vision.RandomErasing()]
        >>> transforms = Compose([py_vision.Decode(),
        ...                       py_transforms.RandomChoice(transforms_list),
        ...                       py_vision.ToTensor()])
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms, input_columns=["image"])
    """

    @check_transforms_list
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to be applied transformation.

        Returns:
            img (PIL image), Transformed image.
        """
        return util.random_choice(img, self.transforms)


class RandomOrder:
    """
    Perform a series of transforms to the input PIL image in a random order.

    Args:
        transforms (list): List of the transformations to apply.

    Examples:
        >>> from mindspore.dataset.transforms.py_transforms import Compose
        >>> transforms_list = [py_vision.RandomHorizontalFlip(0.5),
        ...                    py_vision.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        ...                    py_vision.RandomErasing()]
        >>> transforms = Compose([py_vision.Decode(),
        ...                       py_transforms.RandomOrder(transforms_list),
        ...                       py_vision.ToTensor()])
        >>> image_folder_dataset = image_folder_dataset.map(operations=transforms, input_columns=["image"])
    """

    @check_transforms_list
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL image): Image to apply transformations in a random order.

        Returns:
            img (PIL image), Transformed image.
        """
        return util.random_order(img, self.transforms)
