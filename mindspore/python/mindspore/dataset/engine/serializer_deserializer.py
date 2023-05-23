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
Functions to support dataset serialize and deserialize.
"""
import json
import os

from mindspore import log as logger
from . import datasets as de


def serialize(dataset, json_filepath=""):
    """
    Serialize dataset pipeline into a JSON file.

    Note:
        Complete serialization of Python objects is not currently supported.
        Scenarios that are not supported include data pipelines that use `GeneratorDataset`
        or `map` or `batch` operations that contain custom Python functions.
        For Python objects, serialization operations do not yield the full object content,
        which means that deserialization of the JSON file obtained by serialization may result in errors.
        For example, when serializing the data pipeline of Python user-defined functions,
        a related warning message is reported and the obtained JSON file cannot be deserialized
        into a usable data pipeline.

    Args:
        dataset (Dataset): The starting node.
        json_filepath (str): The filepath where a serialized JSON file will be generated. Default: ``''``.

    Returns:
       Dict, the dictionary contains the serialized dataset graph.

    Raises:
        OSError: Cannot open a file.

    Examples:
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.transforms as transforms
        >>>
        >>> mnist_dataset_dir = "/path/to/mnist_dataset_directory"
        >>> dataset = ds.MnistDataset(mnist_dataset_dir, num_samples=100)
        >>> one_hot_encode = transforms.OneHot(10)  # num_classes is input argument
        >>> dataset = dataset.map(operations=one_hot_encode, input_columns="label")
        >>> dataset = dataset.batch(batch_size=10, drop_remainder=True)
        >>> # serialize it to JSON file
        >>> serialized_data = ds.serialize(dataset, json_filepath="/path/to/mnist_dataset_pipeline.json")
    """
    return dataset.to_json(json_filepath)


def deserialize(input_dict=None, json_filepath=None):
    """
    Construct dataset pipeline from a JSON file produced by dataset serialize function.

    Args:
        input_dict (dict): A Python dictionary containing a serialized dataset graph. Default: ``None``.
        json_filepath (str): A path to the JSON file containing dataset graph.
            User can obtain this file by calling API `mindspore.dataset.serialize()` . Default: ``None``.

    Returns:
        de.Dataset or None if error occurs.

    Raises:
        OSError: Can not open the JSON file.

    Examples:
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.transforms as transforms
        >>>
        >>> mnist_dataset_dir = "/path/to/mnist_dataset_directory"
        >>> dataset = ds.MnistDataset(mnist_dataset_dir, num_samples=100)
        >>> one_hot_encode = transforms.OneHot(10)  # num_classes is input argument
        >>> dataset = dataset.map(operations=one_hot_encode, input_columns="label")
        >>> dataset = dataset.batch(batch_size=10, drop_remainder=True)
        >>>
        >>> # Case 1: to/from JSON file
        >>> serialized_data = ds.serialize(dataset, json_filepath="/path/to/mnist_dataset_pipeline.json")
        >>> deserialized_dataset = ds.deserialize(json_filepath="/path/to/mnist_dataset_pipeline.json")
        >>>
        >>> # Case 2: to/from Python dictionary
        >>> serialized_data = ds.serialize(dataset)
        >>> deserialized_dataset = ds.deserialize(input_dict=serialized_data)
    """

    data = None
    if input_dict:
        data = de.DeserializedDataset(input_dict)

    if json_filepath:
        data = de.DeserializedDataset(json_filepath)
    return data


def expand_path(node_repr, key, val):
    """Convert relative to absolute path."""
    if isinstance(val, list):
        node_repr[key] = [os.path.abspath(file) for file in val]
    else:
        node_repr[key] = os.path.abspath(val)


def show(dataset, indentation=2):
    """
    Write the dataset pipeline graph to logger.info file.

    Args:
        dataset (Dataset): The starting node.
        indentation (int, optional): The indentation used by the JSON print.
            Do not indent if indentation is None. Default: ``2``, indent 2 space.

    Examples:
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.transforms as transforms
        >>>
        >>> mnist_dataset_dir = "/path/to/mnist_dataset_directory"
        >>> dataset = ds.MnistDataset(mnist_dataset_dir, num_samples=100)
        >>> one_hot_encode = transforms.OneHot(10)
        >>> dataset = dataset.map(operations=one_hot_encode, input_columns="label")
        >>> dataset = dataset.batch(batch_size=10, drop_remainder=True)
        >>> ds.show(dataset)
    """

    pipeline = dataset.to_json()
    logger.info(json.dumps(pipeline, indent=indentation))


def compare(pipeline1, pipeline2):
    """
    Compare if two dataset pipelines are the same.

    Args:
        pipeline1 (Dataset): a dataset pipeline.
        pipeline2 (Dataset): a dataset pipeline.

    Returns:
        Whether pipeline1 is equal to pipeline2.

    Examples:
        >>> import mindspore.dataset as ds
        >>>
        >>> pipeline1 = ds.MnistDataset("/path/to/mnist_dataset_directory", num_samples=100)
        >>> pipeline2 = ds.Cifar10Dataset("/path/to/cifar10_dataset_directory", num_samples=100)
        >>> res = ds.compare(pipeline1, pipeline2)
    """

    return pipeline1.to_json() == pipeline2.to_json()
