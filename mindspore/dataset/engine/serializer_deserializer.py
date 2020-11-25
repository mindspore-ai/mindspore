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
Functions to support dataset serialize and deserialize.
"""
import json
import os
import sys

from mindspore import log as logger
from . import datasets as de
from ..vision.utils import Inter, Border
from ..core import config


def serialize(dataset, json_filepath=None):
    """
    Serialize dataset pipeline into a json file.

    Args:
        dataset (Dataset): the starting node.
        json_filepath (str): a filepath where a serialized json file will be generated.

    Returns:
       dict containing the serialized dataset graph.

    Raises:
        OSError cannot open a file

    Examples:
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.transforms.c_transforms as C
        >>> DATA_DIR = "../../data/testMnistData"
        >>> data = ds.MnistDataset(DATA_DIR, 100)
        >>> one_hot_encode = C.OneHot(10)  # num_classes is input argument
        >>> data = data.map(operation=one_hot_encode, input_column_names="label")
        >>> data = data.batch(batch_size=10, drop_remainder=True)
        >>>
        >>> ds.engine.serialize(data, json_filepath="mnist_dataset_pipeline.json")  # serialize it to json file
        >>> serialized_data = ds.engine.serialize(data)  # serialize it to Python dict
    """
    serialized_pipeline = traverse(dataset)
    if json_filepath:
        with open(json_filepath, 'w') as json_file:
            json.dump(serialized_pipeline, json_file, indent=2)
    return serialized_pipeline


def deserialize(input_dict=None, json_filepath=None):
    """
    Construct a de pipeline from a json file produced by de.serialize().

    Args:
        input_dict (dict): a Python dictionary containing a serialized dataset graph
        json_filepath (str): a path to the json file.

    Returns:
        de.Dataset or None if error occurs.

    Raises:
        OSError cannot open a file.

    Examples:
        >>> import mindspore.dataset as ds
        >>> import mindspore.dataset.transforms.c_transforms as C
        >>> DATA_DIR = "../../data/testMnistData"
        >>> data = ds.MnistDataset(DATA_DIR, 100)
        >>> one_hot_encode = C.OneHot(10)  # num_classes is input argument
        >>> data = data.map(operation=one_hot_encode, input_column_names="label")
        >>> data = data.batch(batch_size=10, drop_remainder=True)
        >>>
        >>> # Use case 1: to/from json file
        >>> ds.engine.serialize(data, json_filepath="mnist_dataset_pipeline.json")
        >>> data = ds.engine.deserialize(json_filepath="mnist_dataset_pipeline.json")
        >>> # Use case 2: to/from Python dictionary
        >>> serialized_data = ds.engine.serialize(data)
        >>> data = ds.engine.deserialize(input_dict=serialized_data)

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


def serialize_operations(node_repr, key, val):
    """Serialize tensor op (Python object) to dictionary."""
    if isinstance(val, list):
        node_repr[key] = []
        for op in val:
            node_repr[key].append(op.__dict__)
            # Extracting module and name information from a Python object
            # Example: tensor_op_module is 'minddata.transforms.c_transforms' and tensor_op_name is 'Decode'
            node_repr[key][-1]['tensor_op_name'] = type(op).__name__
            node_repr[key][-1]['tensor_op_module'] = type(op).__module__
    else:
        node_repr[key] = val.__dict__
        node_repr[key]['tensor_op_name'] = type(val).__name__
        node_repr[key]['tensor_op_module'] = type(val).__module__


def serialize_sampler(node_repr, val):
    """Serialize sampler object to dictionary."""
    if val is None:
        node_repr['sampler'] = None
    else:
        node_repr['sampler'] = val.__dict__
        node_repr['sampler']['sampler_module'] = type(val).__module__
        node_repr['sampler']['sampler_name'] = type(val).__name__


def traverse(node):
    """Pre-order traverse the pipeline and capture the information as we go."""
    # Node representation (node_repr) is a Python dictionary that capture and store the
    # dataset pipeline information before dumping it to JSON or other format.
    node_repr = dict()
    node_repr['op_type'] = type(node).__name__
    node_repr['op_module'] = type(node).__module__

    # Start with an empty list of children, will be added later as we traverse this node.
    node_repr["children"] = []

    # Retrieve the information about the current node. It should include arguments
    # passed to the node during object construction.
    node_args = node.get_args()
    for k, v in node_args.items():
        # Store the information about this node into node_repr.
        # Further serialize the object in the arguments if needed.
        if k == 'operations':
            serialize_operations(node_repr, k, v)
        elif k == 'sampler':
            serialize_sampler(node_repr, v)
        elif k == 'padded_sample' and v:
            v1 = {key: value for key, value in v.items() if not isinstance(value, bytes)}
            node_repr[k] = json.dumps(v1, indent=2)
        # return schema json str if its type is mindspore.dataset.Schema
        elif k == 'schema' and isinstance(v, de.Schema):
            node_repr[k] = v.to_json()
        elif k in set(['schema', 'dataset_files', 'dataset_dir', 'schema_file_path']):
            expand_path(node_repr, k, v)
        elif k == "num_parallel_workers" and v is None:
            node_repr[k] = config.get_num_parallel_workers()
        else:
            node_repr[k] = v

    # If a sampler exists in this node, then the following 4 arguments must be set to None:
    #    num_samples, shard_id, num_shards, shuffle
    # These arguments get moved into the sampler itself, so they are no longer needed to
    # be set at the dataset level.
    # TF Record is a special case because it uses both the dataset and sampler arguments
    # which is not decided until later during tree preparation phase.
    if node_repr['op_type'] != 'TFRecordDataset' and 'sampler' in node_args.keys():
        if 'num_samples' in node_repr.keys():
            node_repr['num_samples'] = None
        if 'shuffle' in node_repr.keys():
            node_repr['shuffle'] = None
        if 'num_shards' in node_repr.keys():
            node_repr['num_shards'] = None
        if 'shard_id' in node_repr.keys():
            node_repr['shard_id'] = None

    # Leaf node doesn't have input attribute.
    if not node.children:
        return node_repr

    # Recursively traverse the child and assign it to the current node_repr['children'].
    for child in node.children:
        node_repr["children"].append(traverse(child))

    return node_repr


def show(dataset, indentation=2):
    """
    Write the dataset pipeline graph onto logger.info.

    Args:
        dataset (Dataset): the starting node.
        indentation (int, optional): indentation used by the json print. Pass None to not indent.
    """

    pipeline = traverse(dataset)
    logger.info(json.dumps(pipeline, indent=indentation))


def compare(pipeline1, pipeline2):
    """
    Compare if two dataset pipelines are the same.

    Args:
        pipeline1 (Dataset): a dataset pipeline.
        pipeline2 (Dataset): a dataset pipeline.
    """

    return traverse(pipeline1) == traverse(pipeline2)


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
    op_module = node['op_module']

    # Get the Python class to be instantiated.
    # Example:
    #  "op_type": "MapDataset",
    #  "op_module": "mindspore.dataset.datasets",
    pyclass = getattr(sys.modules[op_module], dataset_op)

    pyobj = None
    # Find a matching Dataset class and call the constructor with the corresponding args.
    # When a new Dataset class is introduced, another if clause and parsing code needs to be added.
    if dataset_op == 'ImageFolderDataset':
        sampler = construct_sampler(node.get('sampler'))
        pyobj = pyclass(node['dataset_dir'], node.get('num_samples'), node.get('num_parallel_workers'),
                        node.get('shuffle'), sampler, node.get('extensions'),
                        node.get('class_indexing'), node.get('decode'), node.get('num_shards'),
                        node.get('shard_id'))

    elif dataset_op == 'RangeDataset':
        pyobj = pyclass(node['start'], node['stop'], node['step'])

    elif dataset_op == 'ImageFolderDataset':
        pyobj = pyclass(node['dataset_dir'], node['schema'], node.get('distribution'),
                        node.get('column_list'), node.get('num_parallel_workers'),
                        node.get('deterministic_output'), node.get('prefetch_size'),
                        node.get('labels_filename'), node.get('dataset_usage'))

    elif dataset_op == 'MnistDataset':
        sampler = construct_sampler(node.get('sampler'))
        pyobj = pyclass(node['dataset_dir'], node['usage'], node.get('num_samples'), node.get('num_parallel_workers'),
                        node.get('shuffle'), sampler, node.get('num_shards'), node.get('shard_id'))

    elif dataset_op == 'MindDataset':
        sampler = construct_sampler(node.get('sampler'))
        pyobj = pyclass(node['dataset_file'], node.get('columns_list'),
                        node.get('num_parallel_workers'), node.get('seed'), node.get('num_shards'),
                        node.get('shard_id'), sampler)

    elif dataset_op == 'TFRecordDataset':
        shuffle = node.get('shuffle')
        if shuffle is not None and isinstance(shuffle, str):
            shuffle = de.Shuffle(shuffle)
        pyobj = pyclass(node['dataset_files'], node.get('schema'), node.get('column_list'),
                        node.get('num_samples'), node.get('num_parallel_workers'),
                        shuffle, node.get('num_shards'), node.get('shard_id'))

    elif dataset_op == 'ManifestDataset':
        sampler = construct_sampler(node.get('sampler'))
        pyobj = pyclass(node['dataset_file'], node['usage'], node.get('num_samples'),
                        node.get('num_parallel_workers'), node.get('shuffle'), sampler,
                        node.get('class_indexing'), node.get('decode'), node.get('num_shards'),
                        node.get('shard_id'))

    elif dataset_op == 'Cifar10Dataset':
        sampler = construct_sampler(node.get('sampler'))
        pyobj = pyclass(node['dataset_dir'], node['usage'], node.get('num_samples'), node.get('num_parallel_workers'),
                        node.get('shuffle'), sampler, node.get('num_shards'), node.get('shard_id'))

    elif dataset_op == 'Cifar100Dataset':
        sampler = construct_sampler(node.get('sampler'))
        pyobj = pyclass(node['dataset_dir'], node['usage'], node.get('num_samples'), node.get('num_parallel_workers'),
                        node.get('shuffle'), sampler, node.get('num_shards'), node.get('shard_id'))

    elif dataset_op == 'VOCDataset':
        sampler = construct_sampler(node.get('sampler'))
        pyobj = pyclass(node['dataset_dir'], node.get('task'), node.get('mode'), node.get('class_indexing'),
                        node.get('num_samples'), node.get('num_parallel_workers'), node.get('shuffle'),
                        node.get('decode'), sampler, node.get('num_shards'), node.get('shard_id'))

    elif dataset_op == 'CocoDataset':
        sampler = construct_sampler(node.get('sampler'))
        pyobj = pyclass(node['dataset_dir'], node.get('annotation_file'), node.get('task'), node.get('num_samples'),
                        node.get('num_parallel_workers'), node.get('shuffle'), node.get('decode'), sampler,
                        node.get('num_shards'), node.get('shard_id'))

    elif dataset_op == 'CelebADataset':
        sampler = construct_sampler(node.get('sampler'))
        pyobj = pyclass(node['dataset_dir'], node.get('num_parallel_workers'), node.get('shuffle'),
                        node.get('dataset_type'), sampler, node.get('decode'), node.get('extensions'),
                        node.get('num_samples'), sampler, node.get('num_shards'), node.get('shard_id'))

    elif dataset_op == 'GeneratorDataset':
        # Serializing py function can be done using marshal library
        raise RuntimeError(dataset_op + " is not yet supported")

    elif dataset_op == 'RepeatDataset':
        pyobj = de.Dataset().repeat(node.get('count'))

    elif dataset_op == 'SkipDataset':
        pyobj = de.Dataset().skip(node.get('count'))

    elif dataset_op == 'TakeDataset':
        pyobj = de.Dataset().take(node.get('count'))

    elif dataset_op == 'MapDataset':
        tensor_ops = construct_tensor_ops(node.get('operations'))
        pyobj = de.Dataset().map(tensor_ops, node.get('input_columns'), node.get('output_columns'),
                                 node.get('column_order'), node.get('num_parallel_workers'))

    elif dataset_op == 'ShuffleDataset':
        pyobj = de.Dataset().shuffle(node.get('buffer_size'))

    elif dataset_op == 'BatchDataset':
        pyobj = de.Dataset().batch(node['batch_size'], node.get('drop_remainder'))

    elif dataset_op == 'CacheDataset':
        # Member function cache() is not defined in class Dataset yet.
        raise RuntimeError(dataset_op + " is not yet supported.")

    elif dataset_op == 'FilterDataset':
        # Member function filter() is not defined in class Dataset yet.
        raise RuntimeError(dataset_op + " is not yet supported.")

    elif dataset_op == 'TakeDataset':
        # Member function take() is not defined in class Dataset yet.
        raise RuntimeError(dataset_op + " is not yet supported.")

    elif dataset_op == 'ZipDataset':
        # Create ZipDataset instance, giving dummy input dataset that will be overrided in the caller.
        pyobj = de.ZipDataset((de.Dataset(), de.Dataset()))

    elif dataset_op == 'ConcatDataset':
        # Create ConcatDataset instance, giving dummy input dataset that will be overrided in the caller.
        pyobj = de.ConcatDataset((de.Dataset(), de.Dataset()))

    elif dataset_op == 'RenameDataset':
        pyobj = de.Dataset().rename(node['input_columns'], node['output_columns'])

    elif dataset_op == 'ProjectDataset':
        pyobj = de.Dataset().project(node['columns'])

    elif dataset_op == 'TransferDataset':
        pyobj = de.Dataset().to_device()

    else:
        raise RuntimeError(dataset_op + " is not yet supported by ds.engine.deserialize().")

    return pyobj


def construct_sampler(in_sampler):
    """Instantiate Sampler object based on the information from dictionary['sampler']"""
    sampler = None
    if in_sampler is not None:
        sampler_name = in_sampler['sampler_name']
        sampler_module = in_sampler['sampler_module']
        sampler_class = getattr(sys.modules[sampler_module], sampler_name)
        if sampler_name == 'DistributedSampler':
            sampler = sampler_class(in_sampler['num_shards'], in_sampler['shard_id'], in_sampler.get('shuffle'))
        elif sampler_name == 'PKSampler':
            sampler = sampler_class(in_sampler['num_val'], in_sampler.get('num_class'), in_sampler('shuffle'))
        elif sampler_name == 'RandomSampler':
            sampler = sampler_class(in_sampler.get('replacement'), in_sampler.get('num_samples'))
        elif sampler_name == 'SequentialSampler':
            sampler = sampler_class()
        elif sampler_name == 'SubsetRandomSampler':
            sampler = sampler_class(in_sampler['indices'])
        elif sampler_name == 'WeightedRandomSampler':
            sampler = sampler_class(in_sampler['weights'], in_sampler['num_samples'], in_sampler.get('replacement'))
        else:
            raise ValueError("Sampler type is unknown: {}.".format(sampler_name))

    return sampler


def construct_tensor_ops(operations):
    """Instantiate tensor op object(s) based on the information from dictionary['operations']"""
    result = []
    for op in operations:
        op_module = op['tensor_op_module']
        op_name = op['tensor_op_name']
        op_class = getattr(sys.modules[op_module], op_name)

        if op_name == 'Decode':
            result.append(op_class(op.get('rgb')))

        elif op_name == 'Normalize':
            result.append(op_class(op['mean'], op['std']))

        elif op_name == 'RandomCrop':
            result.append(op_class(op['size'], op.get('padding'), op.get('pad_if_needed'),
                                   op.get('fill_value'), Border(op.get('padding_mode'))))

        elif op_name == 'RandomHorizontalFlip':
            result.append(op_class(op.get('prob')))

        elif op_name == 'RandomVerticalFlip':
            result.append(op_class(op.get('prob')))

        elif op_name == 'Resize':
            result.append(op_class(op['size'], Inter(op.get('interpolation'))))

        elif op_name == 'RandomResizedCrop':
            result.append(op_class(op['size'], op.get('scale'), op.get('ratio'),
                                   Inter(op.get('interpolation')), op.get('max_attempts')))

        elif op_name == 'CenterCrop':
            result.append(op_class(op['size']))

        elif op_name == 'RandomColorAdjust':
            result.append(op_class(op.get('brightness'), op.get('contrast'), op.get('saturation'),
                                   op.get('hue')))

        elif op_name == 'RandomRotation':
            result.append(op_class(op['degree'], op.get('resample'), op.get('expand'),
                                   op.get('center'), op.get('fill_value')))

        elif op_name == 'Rescale':
            result.append(op_class(op['rescale'], op['shift']))

        elif op_name == 'RandomResize':
            result.append(op_class(op['size']))

        elif op_name == 'TypeCast':
            result.append(op_class(op['data_type']))

        elif op_name == 'HWC2CHW':
            result.append(op_class())

        elif op_name == 'CHW2HWC':
            raise ValueError("Tensor op is not supported: {}.".format(op_name))

        elif op_name == 'OneHot':
            result.append(op_class(op['num_classes']))

        elif op_name == 'RandomCropDecodeResize':
            result.append(op_class(op['size'], op.get('scale'), op.get('ratio'),
                                   Inter(op.get('interpolation')), op.get('max_attempts')))

        elif op_name == 'Pad':
            result.append(op_class(op['padding'], op['fill_value'], Border(op['padding_mode'])))

        else:
            raise ValueError("Tensor op name is unknown: {}.".format(op_name))

    return result
