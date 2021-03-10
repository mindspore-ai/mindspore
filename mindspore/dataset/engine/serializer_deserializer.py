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
import sys

import mindspore.common.dtype as mstype
from mindspore import log as logger
from . import datasets as de
from ..vision.utils import Inter, Border, ImageBatchFormat


def serialize(dataset, json_filepath=""):
    """
    Serialize dataset pipeline into a json file.

    Currently some python objects are not supported to be serialized.
    For python function serialization of map operator, de.serialize will only return its function name.

    Args:
        dataset (Dataset): the starting node.
        json_filepath (str): a filepath where a serialized json file will be generated.

    Returns:
       dict containing the serialized dataset graph.

    Raises:
        OSError cannot open a file

    Examples:
        >>> dataset = ds.MnistDataset(mnist_dataset_dir, 100)
        >>> one_hot_encode = c_transforms.OneHot(10)  # num_classes is input argument
        >>> dataset = dataset.map(operation=one_hot_encode, input_column_names="label")
        >>> dataset = dataset.batch(batch_size=10, drop_remainder=True)
        >>> # serialize it to json file
        >>> ds.engine.serialize(dataset, json_filepath="/path/to/mnist_dataset_pipeline.json")
        >>> serialized_data = ds.engine.serialize(dataset)  # serialize it to Python dict
    """
    return dataset.to_json(json_filepath)


def deserialize(input_dict=None, json_filepath=None):
    """
    Construct a de pipeline from a json file produced by de.serialize().

    Currently python function deserialization of map operator are not supported.

    Args:
        input_dict (dict): a Python dictionary containing a serialized dataset graph
        json_filepath (str): a path to the json file.

    Returns:
        de.Dataset or None if error occurs.

    Raises:
        OSError cannot open a file.

    Examples:
        >>> dataset = ds.MnistDataset(mnist_dataset_dir, 100)
        >>> one_hot_encode = c_transforms.OneHot(10)  # num_classes is input argument
        >>> dataset = dataset.map(operation=one_hot_encode, input_column_names="label")
        >>> dataset = dataset.batch(batch_size=10, drop_remainder=True)
        >>> # Use case 1: to/from json file
        >>> ds.engine.serialize(dataset, json_filepath="/path/to/mnist_dataset_pipeline.json")
        >>> dataset = ds.engine.deserialize(json_filepath="/path/to/mnist_dataset_pipeline.json")
        >>> # Use case 2: to/from Python dictionary
        >>> serialized_data = ds.engine.serialize(dataset)
        >>> dataset = ds.engine.deserialize(input_dict=serialized_data)

    """
    data = None
    if input_dict:
        data = construct_pipeline(input_dict)

    if json_filepath:
        dict_pipeline = dict()
        with open(json_filepath, 'r') as json_file:
            dict_pipeline = json.load(json_file)
            data = construct_pipeline(dict_pipeline)

    return data


def expand_path(node_repr, key, val):
    """Convert relative to absolute path."""
    if isinstance(val, list):
        node_repr[key] = [os.path.abspath(file) for file in val]
    else:
        node_repr[key] = os.path.abspath(val)


def show(dataset, indentation=2):
    """
    Write the dataset pipeline graph onto logger.info.

    Args:
        dataset (Dataset): the starting node.
        indentation (int, optional): indentation used by the json print. Pass None to not indent.
    """

    pipeline = dataset.to_json()
    logger.info(json.dumps(pipeline, indent=indentation))


def compare(pipeline1, pipeline2):
    """
    Compare if two dataset pipelines are the same.

    Args:
        pipeline1 (Dataset): a dataset pipeline.
        pipeline2 (Dataset): a dataset pipeline.
    """

    return pipeline1.to_json() == pipeline2.to_json()


def construct_pipeline(node):
    """Construct the Python Dataset objects by following the dictionary deserialized from json file."""
    op_type = node.get('op_type')
    if not op_type:
        raise ValueError("op_type field in the json file can't be None.")

    # Instantiate Python Dataset object based on the current dictionary element
    dataset = create_node(node)
    # Initially it is not connected to any other object.
    dataset.children = []

    # Construct the children too and add edge between the children and parent.
    for child in node['children']:
        dataset.children.append(construct_pipeline(child))

    return dataset


def create_node(node):
    """Parse the key, value in the node dictionary and instantiate the Python Dataset object"""
    logger.info('creating node: %s', node['op_type'])
    dataset_op = node['op_type']
    op_module = "mindspore.dataset"

    # Get the Python class to be instantiated.
    # Example:
    #  "op_type": "MapDataset",
    #  "op_module": "mindspore.dataset.datasets",
    if node.get("children"):
        pyclass = getattr(sys.modules[op_module], "Dataset")
    else:
        pyclass = getattr(sys.modules[op_module], dataset_op)

    pyobj = None
    # Find a matching Dataset class and call the constructor with the corresponding args.
    # When a new Dataset class is introduced, another if clause and parsing code needs to be added.
    # Dataset Source Ops (in alphabetical order)
    pyobj = create_dataset_node(pyclass, node, dataset_op)
    if not pyobj:
        # Dataset Ops (in alphabetical order)
        pyobj = create_dataset_operation_node(node, dataset_op)

    return pyobj


def create_dataset_node(pyclass, node, dataset_op):
    """Parse the key, value in the dataset node dictionary and instantiate the Python Dataset object"""
    pyobj = None
    if dataset_op == 'CelebADataset':
        sampler = construct_sampler(node.get('sampler'))
        num_samples = check_and_replace_input(node.get('num_samples'), 0, None)
        pyobj = pyclass(node['dataset_dir'], node.get('num_parallel_workers'), node.get('shuffle'), node.get('usage'),
                        sampler, node.get('decode'), node.get('extensions'), num_samples, node.get('num_shards'),
                        node.get('shard_id'))

    elif dataset_op == 'Cifar10Dataset':
        sampler = construct_sampler(node.get('sampler'))
        num_samples = check_and_replace_input(node.get('num_samples'), 0, None)
        pyobj = pyclass(node['dataset_dir'], node['usage'], num_samples, node.get('num_parallel_workers'),
                        node.get('shuffle'), sampler, node.get('num_shards'), node.get('shard_id'))

    elif dataset_op == 'Cifar100Dataset':
        sampler = construct_sampler(node.get('sampler'))
        num_samples = check_and_replace_input(node.get('num_samples'), 0, None)
        pyobj = pyclass(node['dataset_dir'], node['usage'], num_samples, node.get('num_parallel_workers'),
                        node.get('shuffle'), sampler, node.get('num_shards'), node.get('shard_id'))

    elif dataset_op == 'ClueDataset':
        shuffle = to_shuffle_mode(node.get('shuffle'))
        if isinstance(shuffle, str):
            shuffle = de.Shuffle(shuffle)
        num_samples = check_and_replace_input(node.get('num_samples'), 0, None)
        pyobj = pyclass(node['dataset_files'], node.get('task'),
                        node.get('usage'), num_samples, node.get('num_parallel_workers'), shuffle,
                        node.get('num_shards'), node.get('shard_id'))

    elif dataset_op == 'CocoDataset':
        sampler = construct_sampler(node.get('sampler'))
        num_samples = check_and_replace_input(node.get('num_samples'), 0, None)
        pyobj = pyclass(node['dataset_dir'], node.get('annotation_file'), node.get('task'), num_samples,
                        node.get('num_parallel_workers'), node.get('shuffle'), node.get('decode'), sampler,
                        node.get('num_shards'), node.get('shard_id'))

    elif dataset_op == 'CSVDataset':
        shuffle = to_shuffle_mode(node.get('shuffle'))
        if isinstance(shuffle, str):
            shuffle = de.Shuffle(shuffle)
        num_samples = check_and_replace_input(node.get('num_samples'), 0, None)
        pyobj = pyclass(node['dataset_files'], node.get('field_delim'),
                        node.get('column_defaults'), node.get('column_names'), num_samples,
                        node.get('num_parallel_workers'), shuffle,
                        node.get('num_shards'), node.get('shard_id'))

    elif dataset_op == 'ImageFolderDataset':
        sampler = construct_sampler(node.get('sampler'))
        num_samples = check_and_replace_input(node.get('num_samples'), 0, None)
        pyobj = pyclass(node['dataset_dir'], num_samples, node.get('num_parallel_workers'),
                        node.get('shuffle'), sampler, node.get('extensions'),
                        node.get('class_indexing'), node.get('decode'), node.get('num_shards'),
                        node.get('shard_id'))

    elif dataset_op == 'ManifestDataset':
        sampler = construct_sampler(node.get('sampler'))
        num_samples = check_and_replace_input(node.get('num_samples'), 0, None)
        pyobj = pyclass(node['dataset_file'], node['usage'], num_samples,
                        node.get('num_parallel_workers'), node.get('shuffle'), sampler,
                        node.get('class_indexing'), node.get('decode'), node.get('num_shards'),
                        node.get('shard_id'))

    elif dataset_op == 'MnistDataset':
        sampler = construct_sampler(node.get('sampler'))
        num_samples = check_and_replace_input(node.get('num_samples'), 0, None)
        pyobj = pyclass(node['dataset_dir'], node['usage'], num_samples, node.get('num_parallel_workers'),
                        node.get('shuffle'), sampler, node.get('num_shards'), node.get('shard_id'))

    elif dataset_op == 'TextFileDataset':
        shuffle = to_shuffle_mode(node.get('shuffle'))
        if isinstance(shuffle, str):
            shuffle = de.Shuffle(shuffle)
        num_samples = check_and_replace_input(node.get('num_samples'), 0, None)
        pyobj = pyclass(node['dataset_files'], num_samples,
                        node.get('num_parallel_workers'), shuffle,
                        node.get('num_shards'), node.get('shard_id'))

    elif dataset_op == 'TFRecordDataset':
        shuffle = to_shuffle_mode(node.get('shuffle'))
        if isinstance(shuffle, str):
            shuffle = de.Shuffle(shuffle)
        num_samples = check_and_replace_input(node.get('num_samples'), 0, None)
        pyobj = pyclass(node['dataset_files'], node.get('schema'), node.get('columns_list'),
                        num_samples, node.get('num_parallel_workers'),
                        shuffle, node.get('num_shards'), node.get('shard_id'))

    elif dataset_op == 'VOCDataset':
        sampler = construct_sampler(node.get('sampler'))
        num_samples = check_and_replace_input(node.get('num_samples'), 0, None)
        pyobj = pyclass(node['dataset_dir'], node.get('task'), node.get('usage'), node.get('class_indexing'),
                        num_samples, node.get('num_parallel_workers'), node.get('shuffle'),
                        node.get('decode'), sampler, node.get('num_shards'), node.get('shard_id'))

    return pyobj


def create_dataset_operation_node(node, dataset_op):
    """Parse the key, value in the dataset operation node dictionary and instantiate the Python Dataset object"""
    pyobj = None
    if dataset_op == 'Batch':
        pyobj = de.Dataset().batch(node['batch_size'], node.get('drop_remainder'))

    elif dataset_op == 'Map':
        tensor_ops = construct_tensor_ops(node.get('operations'))
        pyobj = de.Dataset().map(tensor_ops, node.get('input_columns'), node.get('output_columns'),
                                 node.get('column_order'), node.get('num_parallel_workers'),
                                 False, None, node.get('callbacks'))

    elif dataset_op == 'Project':
        pyobj = de.Dataset().project(node['columns'])

    elif dataset_op == 'Rename':
        pyobj = de.Dataset().rename(node['input_columns'], node['output_columns'])

    elif dataset_op == 'Repeat':
        pyobj = de.Dataset().repeat(node.get('count'))

    elif dataset_op == 'Shuffle':
        pyobj = de.Dataset().shuffle(node.get('buffer_size'))

    elif dataset_op == 'Skip':
        pyobj = de.Dataset().skip(node.get('count'))

    elif dataset_op == 'Take':
        pyobj = de.Dataset().take(node.get('count'))

    elif dataset_op == 'Transfer':
        pyobj = de.Dataset().to_device(node.get('send_epoch_end'), node.get('create_data_info_queue'))

    elif dataset_op == 'Zip':
        # Create ZipDataset instance, giving dummy input dataset that will be overrode in the caller.
        pyobj = de.ZipDataset((de.Dataset(), de.Dataset()))

    else:
        raise RuntimeError(dataset_op + " is not yet supported by ds.engine.deserialize().")

    return pyobj


def construct_sampler(in_sampler):
    """Instantiate Sampler object based on the information from dictionary['sampler']"""
    sampler = None
    if in_sampler is not None:
        if "num_samples" in in_sampler:
            num_samples = check_and_replace_input(in_sampler['num_samples'], 0, None)
        sampler_name = in_sampler['sampler_name']
        sampler_module = "mindspore.dataset"
        sampler_class = getattr(sys.modules[sampler_module], sampler_name)
        if sampler_name == 'DistributedSampler':
            sampler = sampler_class(in_sampler['num_shards'], in_sampler['shard_id'], in_sampler.get('shuffle'))
        elif sampler_name == 'PKSampler':
            sampler = sampler_class(in_sampler['num_val'], in_sampler.get('num_class'), in_sampler('shuffle'))
        elif sampler_name == 'RandomSampler':
            sampler = sampler_class(in_sampler.get('replacement'), num_samples)
        elif sampler_name == 'SequentialSampler':
            sampler = sampler_class(in_sampler.get('start_index'), num_samples)
        elif sampler_name == 'SubsetRandomSampler':
            sampler = sampler_class(in_sampler['indices'], num_samples)
        elif sampler_name == 'WeightedRandomSampler':
            sampler = sampler_class(in_sampler['weights'], num_samples, in_sampler.get('replacement'))
        else:
            raise ValueError("Sampler type is unknown: {}.".format(sampler_name))
    if in_sampler.get("child_sampler"):
        for child in in_sampler["child_sampler"]:
            sampler.add_child(construct_sampler(child))

    return sampler


def construct_tensor_ops(operations):
    """Instantiate tensor op object(s) based on the information from dictionary['operations']"""
    result = []
    for op in operations:
        op_name = op.get('tensor_op_name')
        op_params = op.get('tensor_op_params')

        if op.get('is_python_front_end_op'):  # check if it's a py_transform op
            raise NotImplementedError("python function is not yet supported by de.deserialize().")

        if op_name == "HwcToChw": op_name = "HWC2CHW"
        if op_name == "UniformAug": op_name = "UniformAugment"
        op_module_vis = sys.modules["mindspore.dataset.vision.c_transforms"]
        op_module_trans = sys.modules["mindspore.dataset.transforms.c_transforms"]

        if hasattr(op_module_vis, op_name):
            op_class = getattr(op_module_vis, op_name, None)
        elif hasattr(op_module_trans, op_name[:-2]):
            op_name = op_name[:-2]  # to remove op from the back of the name
            op_class = getattr(op_module_trans, op_name, None)
        else:
            raise RuntimeError(op_name + " is not yet supported by deserialize().")

        if op_params is None:  # If no parameter is specified, call it directly
            result.append(op_class())
        else:
            # Input parameter type cast
            for key, val in op_params.items():
                if key in ['center', 'fill_value']:
                    op_params[key] = tuple(val)
                elif key in ['interpolation', 'resample']:
                    op_params[key] = Inter(to_interpolation_mode(val))
                elif key in ['padding_mode']:
                    op_params[key] = Border(to_border_mode(val))
                elif key in ['data_type']:
                    op_params[key] = to_mstype(val)
                elif key in ['image_batch_format']:
                    op_params[key] = to_image_batch_format(val)
                elif key in ['policy']:
                    op_params[key] = to_policy(val)
                elif key in ['transform', 'transforms']:
                    op_params[key] = construct_tensor_ops(val)

            result.append(op_class(**op_params))
    return result


def to_policy(op_list):
    policy_tensor_ops = []
    for policy_list in op_list:
        sub_policy_tensor_ops = []
        for policy_item in policy_list:
            sub_policy_tensor_ops.append(
                (construct_tensor_ops(policy_item.get('tensor_op')), policy_item.get('prob')))
        policy_tensor_ops.append(sub_policy_tensor_ops)
    return policy_tensor_ops


def to_shuffle_mode(shuffle):
    if shuffle == 2: return "global"
    if shuffle == 1: return "file"
    return False


def to_interpolation_mode(inter):
    return {
        0: Inter.LINEAR,
        1: Inter.NEAREST,
        2: Inter.CUBIC,
        3: Inter.AREA
    }[inter]


def to_border_mode(border):
    return {
        0: Border.CONSTANT,
        1: Border.EDGE,
        2: Border.REFLECT,
        3: Border.SYMMETRIC
    }[border]


def to_mstype(data_type):
    return {
        "bool": mstype.bool_,
        "int8": mstype.int8,
        "int16": mstype.int16,
        "int32": mstype.int32,
        "int64": mstype.int64,
        "uint8": mstype.uint8,
        "uint16": mstype.uint16,
        "uint32": mstype.uint32,
        "uint64": mstype.uint64,
        "float16": mstype.float16,
        "float32": mstype.float32,
        "float64": mstype.float64,
        "string": mstype.string
    }[data_type]


def to_image_batch_format(image_batch_format):
    return {
        0: ImageBatchFormat.NHWC,
        1: ImageBatchFormat.NCHW
    }[image_batch_format]


def check_and_replace_input(input_value, expect, replace):
    return replace if input_value == expect else input_value
