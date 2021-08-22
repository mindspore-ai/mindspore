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
        Currently some Python objects are not supported to be serialized.
        For Python function serialization of map operator, de.serialize will only return its function name.

    Args:
        dataset (Dataset): The starting node.
        json_filepath (str): The filepath where a serialized JSON file will be generated.

    Returns:
       Dict, The dictionary contains the serialized dataset graph.

    Raises:
        OSError: Can not open a file

    Examples:
        >>> dataset = ds.MnistDataset(mnist_dataset_dir, 100)
        >>> one_hot_encode = c_transforms.OneHot(10)  # num_classes is input argument
        >>> dataset = dataset.map(operation=one_hot_encode, input_column_names="label")
        >>> dataset = dataset.batch(batch_size=10, drop_remainder=True)
        >>> # serialize it to JSON file
        >>> ds.engine.serialize(dataset, json_filepath="/path/to/mnist_dataset_pipeline.json")
        >>> serialized_data = ds.engine.serialize(dataset)  # serialize it to Python dict
    """
    return dataset.to_json(json_filepath)


def deserialize(input_dict=None, json_filepath=None):
    """
    Construct a de pipeline from a JSON file produced by de.serialize().

    Note:
        Currently Python function deserialization of map operator are not supported.

    Args:
        input_dict (dict): A Python dictionary containing a serialized dataset graph.
        json_filepath (str): A path to the JSON file.

    Returns:
        de.Dataset or None if error occurs.

    Raises:
        OSError: Can not open the JSON file.

    Examples:
        >>> dataset = ds.MnistDataset(mnist_dataset_dir, 100)
        >>> one_hot_encode = c_transforms.OneHot(10)  # num_classes is input argument
        >>> dataset = dataset.map(operation=one_hot_encode, input_column_names="label")
        >>> dataset = dataset.batch(batch_size=10, drop_remainder=True)
        >>> # Use case 1: to/from JSON file
        >>> ds.engine.serialize(dataset, json_filepath="/path/to/mnist_dataset_pipeline.json")
        >>> dataset = ds.engine.deserialize(json_filepath="/path/to/mnist_dataset_pipeline.json")
        >>> # Use case 2: to/from Python dictionary
        >>> serialized_data = ds.engine.serialize(dataset)
        >>> dataset = ds.engine.deserialize(input_dict=serialized_data)

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
            Do not indent if indentation is None.

    Examples:
        >>> dataset = ds.MnistDataset(mnist_dataset_dir, 100)
        >>> one_hot_encode = c_transforms.OneHot(10)
        >>> dataset = dataset.map(operation=one_hot_encode, input_column_names="label")
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
        >>> pipeline1 = ds.MnistDataset(mnist_dataset_dir, 100)
        >>> pipeline2 = ds.Cifar10Dataset(cifar_dataset_dir, 100)
        >>> ds.compare(pipeline1, pipeline2)
    """

    return pipeline1.to_json() == pipeline2.to_json()
