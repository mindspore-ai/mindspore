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
Testing dataset serialize and deserialize in DE
"""
import filecmp
import glob
import json
import os

import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms as py_transforms
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore import log as logger
from ..dataset.util import config_get_set_num_parallel_workers, config_get_set_seed


def test_serdes_pyvision(remove_json_files=True):
    """
    Feature: Serialize and Deserialize Support
    Description: Test serialize and deserialize on pipelines with Python vision ops
    Expectation: Serialized versus Deserialized+reserialized pipeline output verified
    """
    data_dir = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
    schema_file = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

    original_seed = config_get_set_seed(1)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    data1 = ds.TFRecordDataset(data_dir, schema_file, columns_list=["image", "label"], shuffle=False)
    transforms1 = [
        py_vision.Decode(),
        py_vision.CenterCrop([32, 32])
    ]
    transforms2 = [
        py_vision.RandomColorAdjust(),
        py_vision.FiveCrop(1),
        py_vision.Grayscale()
    ]
    data1 = data1.map(operations=py_transforms.Compose(transforms1), input_columns=["image"])
    data1 = data1.map(operations=py_transforms.RandomApply(transforms2), input_columns=["image"])
    util_check_serialize_deserialize_file(data1, "depr_pyvision_dataset_pipeline", remove_json_files)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

    if remove_json_files:
        delete_json_files("depr_pyvision_dataset_pipeline")


def test_serdes_pyfunc(remove_json_files=True):
    """
    Feature: Serialize and Deserialize Support
    Description: Test serialize and deserialize on pipelines with Python functions
    Expectation: Serialized versus Deserialized+reserialized pipeline output verified
    """
    data_dir = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
    schema_file = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

    original_seed = config_get_set_seed(1)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    data2 = ds.TFRecordDataset(data_dir, schema_file, columns_list=["image", "label"], shuffle=False)
    data2 = data2.map(operations=(lambda x, y, z: (
        np.array(x).flatten().reshape(10, 39),
        np.array(y).flatten().reshape(10, 39),
        np.array(z).flatten().reshape(10, 1)
    )))
    ds.serialize(data2, "pyfunc_dataset_pipeline.json")
    assert validate_jsonfile("pyfunc_dataset_pipeline.json") is True

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

    if remove_json_files:
        delete_json_files("depr_pyfunc_dataset_pipeline")


def test_serdes_inter_mixed_map(remove_json_files=True):
    """
    Feature: Serialize and Deserialize Support
    Description: Test serialize and deserialize on pipelines in which each map op has Python ops or C++ ops
    Expectation: Serialized versus Deserialized+reserialized pipeline output verified
    """
    data_dir = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
    schema_file = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

    original_seed = config_get_set_seed(1)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    data1 = ds.TFRecordDataset(data_dir, schema_file, columns_list=["image", "label"], shuffle=False)
    # The following map op uses Python ops
    data1 = data1.map(operations=[py_vision.Decode(), py_vision.CenterCrop([24, 24])], input_columns=["image"])
    # The following map op uses Python ops
    data1 = data1.map(operations=[py_vision.ToTensor(), py_vision.ToPIL()], input_columns=["image"])
    # The following map op uses C++ ops
    data1 = data1.map(operations=[c_vision.HorizontalFlip(), c_vision.VerticalFlip()], input_columns=["image"])
    # The following map op uses Python ops
    data1 = data1.map(operations=[py_vision.ToPIL(), py_vision.FiveCrop((18, 22))], input_columns=["image"])

    util_check_serialize_deserialize_file(data1, "depr_inter_mixed_map_pipeline", remove_json_files)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

    if remove_json_files:
        delete_json_files("depr_inter_mixed_map_pipeline")


def test_serdes_intra_mixed_py2c_map(remove_json_files=True):
    """
    Feature: Serialize and Deserialize Support
    Description: Test serialize and deserialize on pipelines in which each map op has a mix of Python ops
        then C++ ops
    Expectation: Serialized versus Deserialized+reserialized pipeline output verified
    """
    data_dir = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
    schema_file = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

    original_seed = config_get_set_seed(1)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    data1 = ds.TFRecordDataset(data_dir, schema_file, columns_list=["image", "label"], shuffle=False)
    transforms_list = [py_vision.Decode(),
                       py_vision.CenterCrop([24, 24]),
                       py_vision.ToTensor(),
                       py_vision.Normalize([0.48, 0.45, 0.40], [0.22, 0.22, 0.22]),
                       c_vision.RandomHorizontalFlip(),
                       c_vision.VerticalFlip()]
    data1 = data1.map(operations=transforms_list, input_columns=["image"])
    data2 = util_check_serialize_deserialize_file(data1, "depr_intra_mixed_py2c_map_pipeline", False)

    num_itr = 0
    # Iterate and compare the data in the original pipeline (data1) against the deserialized pipeline (data2)
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1['image'], item2['image'])
        num_itr += 1
    assert num_itr == 3

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

    if remove_json_files:
        delete_json_files("depr_intra_mixed_py2c_map_pipeline")


def test_serdes_intra_mixed_c2py_map(remove_json_files=True):
    """
    Feature: Serialize and Deserialize Support
    Description: Test serialize and deserialize on pipelines in which each map op has a mix of C++ ops
        then Python ops
    Expectation: Serialized versus Deserialized+reserialized pipeline output verified
    """
    data_dir = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
    schema_file = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

    original_seed = config_get_set_seed(1)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    data1 = ds.TFRecordDataset(data_dir, schema_file, columns_list=["image", "label"], shuffle=False)
    transforms_list = [c_vision.Decode(),
                       c_vision.RandomSolarize((0, 127)),
                       py_vision.ToPIL(),
                       py_vision.CenterCrop([64, 64])]
    data1 = data1.map(operations=transforms_list, input_columns=["image"])
    data2 = util_check_serialize_deserialize_file(data1, "depr_intra_mixed_c2py_map_pipeline", False)

    num_itr = 0
    # Iterate and compare the data in the original pipeline (data1) against the deserialized pipeline (data2)
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1['image'], item2['image'])
        num_itr += 1
    assert num_itr == 3

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

    if remove_json_files:
        delete_json_files("depr_intra_mixed_c2py_map_pipeline")


def util_check_serialize_deserialize_file(data_orig, filename, remove_json_files):
    """
    Utility function for testing serdes files. It is to check if a json file is indeed created with correct name
    after serializing and if it remains the same after repeatedly saving and loading.
    :param data_orig: original data pipeline to be serialized
    :param filename: filename to be saved as json format
    :param remove_json_files: whether to remove the json file after testing
    :return: The data pipeline after serializing and deserializing using the original pipeline
    """
    file1 = filename + ".json"
    file2 = filename + "_1.json"
    ds.serialize(data_orig, file1)
    assert validate_jsonfile(file1) is True
    assert validate_jsonfile("wrong_name.json") is False

    data_changed = ds.deserialize(json_filepath=file1)
    ds.serialize(data_changed, file2)
    assert validate_jsonfile(file2) is True
    assert filecmp.cmp(file1, file2, shallow=False)

    # Remove the generated json file
    if remove_json_files:
        delete_json_files(filename)
    return data_changed


def validate_jsonfile(filepath):
    try:
        file_exist = os.path.exists(filepath)
        with open(filepath, 'r') as jfile:
            loaded_json = json.load(jfile)
    except IOError:
        return False
    return file_exist and isinstance(loaded_json, dict)


def delete_json_files(filename):
    file_list = glob.glob(filename + '.json') + glob.glob(filename + '_1.json')
    for f in file_list:
        try:
            os.remove(f)
        except IOError:
            logger.info("Error while deleting: {}".format(f))


if __name__ == '__main__':
    test_serdes_pyvision()
    test_serdes_pyfunc()
    test_serdes_inter_mixed_map()
    test_serdes_intra_mixed_py2c_map()
    test_serdes_intra_mixed_c2py_map()
