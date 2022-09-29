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
import pytest

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as c_transforms
import mindspore.dataset.transforms.py_transforms as py_transforms
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore import log as logger
from mindspore.dataset.vision import Border, Inter
from ..dataset.util import config_get_set_num_parallel_workers, config_get_set_seed


def test_serdes_random_crop():
    """
    Feature: Serialize and Deserialize Support
    Description: Test serialize and deserialize on pipeline with RandomCrop C++ op
    Expectation: Output verified for multiple deserialized pipelines
    """
    logger.info("test_random_crop")
    data_dir = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
    schema_dir = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

    original_seed = config_get_set_seed(1)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # First dataset
    data1 = ds.TFRecordDataset(data_dir, schema_dir, columns_list=["image"])
    decode_op = c_vision.Decode()
    # Test fill_value with tuple
    random_crop_op = c_vision.RandomCrop([1800, 2400], [2, 2, 2, 2],
                                         fill_value=(0, 124, 255))
    data1 = data1.map(operations=[decode_op, random_crop_op], input_columns="image")

    # Serializing into Python dictionary
    ds1_dict = ds.serialize(data1)
    # Serializing into json object
    _ = json.dumps(ds1_dict, indent=2)

    # Reconstruct dataset pipeline from its serialized form
    data1_1 = ds.deserialize(input_dict=ds1_dict)

    # Second dataset
    data2 = ds.TFRecordDataset(data_dir, schema_dir, columns_list=["image"])
    data2 = data2.map(operations=decode_op, input_columns="image")

    for item1, item1_1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                                     data1_1.create_dict_iterator(num_epochs=1, output_numpy=True),
                                     data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1['image'], item1_1['image'])
        _ = item2["image"]

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_serdes_random_rotation():
    """
    Feature: Serialize and Deserialize Support
    Description: Test serialize and deserialize on pipeline with RandomRotation Python op
    Expectation: Output verified for multiple deserialized pipelines
    """
    logger.info("test_random_rotation")
    data_dir = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
    schema_dir = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

    original_seed = config_get_set_seed(1)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # First dataset
    data1 = ds.TFRecordDataset(data_dir, schema_dir, columns_list=["image"])
    decode_op = py_vision.Decode()
    # Test fill_value with tuple
    random_rotation_op = py_vision.RandomRotation((90, 90), expand=True, resample=Inter.BILINEAR,
                                                  center=(50, 50), fill_value=(0, 1, 2))
    data1 = data1.map(operations=[decode_op, random_rotation_op], input_columns="image")

    # Serializing into Python dictionary
    ds1_dict = ds.serialize(data1)
    # Serializing into json object
    _ = json.dumps(ds1_dict, indent=2)

    # Reconstruct dataset pipeline from its serialized form
    data1_1 = ds.deserialize(input_dict=ds1_dict)

    # Second dataset
    data2 = ds.TFRecordDataset(data_dir, schema_dir, columns_list=["image"])
    data2 = data2.map(operations=decode_op, input_columns="image")

    for item1, item1_1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                                     data1_1.create_dict_iterator(num_epochs=1, output_numpy=True),
                                     data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1['image'], item1_1['image'])
        _ = item2["image"]

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


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


def test_serdes_pyfunc_exception(remove_json_files=True):
    """
    Feature: Serialize and Deserialize Support
    Description: Test serialize on pipeline with user-defined Python function
    Expectation: Exception is raised as expected
    """
    data_dir = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
    schema_file = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

    data1 = ds.TFRecordDataset(data_dir, schema_file, columns_list=["image", "label"], shuffle=False)
    data1 = data1.map(operations=(lambda x, y, z: (
        np.array(x).flatten().reshape(10, 39),
        np.array(y).flatten().reshape(10, 39),
        np.array(z).flatten().reshape(10, 1)
    )))

    # Note: Serialization of a dataset pipeline with Python UDF is not supported, and
    #       it is not valid to deserialize the JSON output nor re-serialize the deserialize output.
    ds.serialize(data1, "depr_pyfunc_dataset_pipeline.json")
    data2 = ds.deserialize(input_dict="depr_pyfunc_dataset_pipeline.json")

    with pytest.raises(RuntimeError) as error_info:
        ds.serialize(data2, "depr_pyfunc_dataset_pipeline2.json")
    assert "Failed to find key 'tensor_op_params' in PyFuncOp' JSON file or input dict" in str(error_info.value)

    if remove_json_files:
        delete_json_files("depr_pyfunc_dataset_pipeline")


def test_serdes_pyfunc_exception2(remove_json_files=True):
    """
    Feature: Serialize and Deserialize Support
    Description: Test serialize on pipeline with user-defined Python function
    Expectation: Exception is raised as expected
    """

    def chwtohwc(x):
        """ CHW to HWC """
        return x.transpose(1, 2, 0)

    data_dir = "../data/dataset/testPK/data"
    data1 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False, num_samples=5)

    image_ops1 = [c_vision.RandomCropDecodeResize(250),
                  py_vision.ToPIL(),
                  py_vision.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                  py_vision.ToTensor(),
                  chwtohwc,
                  c_vision.RandomHorizontalFlip(prob=0.5)]

    data1 = data1.map(operations=image_ops1, input_columns="image", num_parallel_workers=8)

    # Perform simple validation for data pipeline
    num = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num += 1
    assert num == 5

    # Note: Serialization of a dataset pipeline with Python UDF is not supported, and
    #       it is not valid to deserialize the JSON output nor re-serialize the deserialize output.
    ds.serialize(data1, "depr_pyfunc2_dataset_pipeline.json")
    data2 = ds.deserialize(input_dict="depr_pyfunc2_dataset_pipeline.json")

    with pytest.raises(AttributeError) as error_info:
        ds.serialize(data2, "depr_pyfunc2_dataset_pipeline2.json")
    assert "no attribute 'chwtohwc'" in str(error_info.value)

    if remove_json_files:
        delete_json_files("depr_pyfunc2_dataset_pipeline")


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


def test_serdes_inter_mixed_enum_parms_map(remove_json_files=True):
    """
    Feature: Serialize and Deserialize Support
    Description: Test serialize and deserialize on pipelines in which each map op has the same
        implementation (Python or C++) of ops, for which the ops have parameters with enumerated types
    Expectation: Serialized versus Deserialized+reserialized pipeline output verified
    """
    data_dir = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
    schema_file = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

    original_seed = config_get_set_seed(26)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    data1 = ds.TFRecordDataset(data_dir, schema_file, columns_list=["image", "label"], shuffle=False)
    # The following map op uses Python implementation of ops
    data1 = data1.map(operations=[py_vision.Decode(),
                                  py_vision.Resize((250, 300), interpolation=Inter.LINEAR),
                                  py_vision.RandomCrop(size=250, padding=[100, 100, 100, 100],
                                                       padding_mode=Border.EDGE),
                                  py_vision.RandomRotation((0, 90), expand=True, resample=Inter.BILINEAR,
                                                           center=(50, 50), fill_value=150)],
                      input_columns=["image"])
    # The following map op uses C++ implementation of ToTensor op
    data1 = data1.map(operations=[py_vision.ToTensor()], input_columns=["image"])
    # The following map op uses C++ implementation of ops
    data1 = data1.map(operations=[c_vision.Pad(padding=[100, 100, 100, 100], fill_value=150,
                                               padding_mode=Border.REFLECT)],
                      input_columns=["image"])
    # The following map op uses Python implementation of ops
    data1 = data1.map(operations=[py_vision.ToPIL(),
                                  py_vision.RandomPerspective(0.3, 1.0, Inter.LINEAR),
                                  py_vision.RandomAffine(degrees=15, translate=(-0.1, 0.1, 0, 0), scale=(0.9, 1.1),
                                                         resample=Inter.NEAREST)],
                      input_columns=["image"])

    util_check_serialize_deserialize_file(data1, "depr_inter_mixed_map_pipeline", remove_json_files)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

    if remove_json_files:
        delete_json_files("depr_inter_mixed_enum_parms_map_pipeline")


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


def test_serdes_uniform_augment(remove_json_files=True):
    """
    Feature: Serialize and Deserialize Support
    Description: Test serialize and deserialize on pipeline with UniformAugment op
    Expectation: Serialized versus Deserialized+reserialized pipeline output verified
    """
    original_seed = config_get_set_seed(1)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    data_dir = "../data/dataset/testPK/data"
    data = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)

    transforms_ua = [c_vision.RandomHorizontalFlip(),
                     c_vision.RandomVerticalFlip(),
                     c_vision.RandomColor(),
                     c_vision.RandomSharpness(),
                     c_vision.Invert(),
                     c_vision.AutoContrast(),
                     c_vision.Equalize()]
    transforms_all = [c_vision.Decode(), c_vision.Resize(size=[224, 224]),
                      c_vision.UniformAugment(transforms=transforms_ua, num_ops=5)]
    data = data.map(operations=transforms_all, input_columns="image", num_parallel_workers=1)
    util_check_serialize_deserialize_file(data, "depr_uniform_augment_pipeline", remove_json_files)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_serdes_complex1_pipeline(remove_json_files=True):
    """
    Feature: Serialize and Deserialize Support
    Description: Test serialize on complex pipeline with mix of C++ implementation ops and Python implementation ops
    Expectation: Serialized versus Deserialized+reserialized pipeline output verified
    """
    original_seed = config_get_set_seed(1)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    data_dir = "../data/dataset/testPK/data"
    data1 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)

    type_cast_op = c_transforms.TypeCast(mstype.int32)
    image_ops1 = [c_vision.RandomCropDecodeResize(250),
                  py_vision.ToPIL(),
                  py_vision.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                  py_vision.RandomHorizontalFlip(prob=0.5)]

    image_ops2 = [c_vision.RandomColorAdjust(),
                  c_vision.RandomSharpness(),
                  c_vision.RandomVerticalFlip(),
                  c_vision.Rescale(0.5, 1.0),
                  c_vision.Normalize(mean=[121.0, 115.0, 100.0], std=[70.0, 68.0, 71.0]),
                  c_vision.HWC2CHW()]

    data1 = data1.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    data1 = data1.map(operations=image_ops1, input_columns="image", num_parallel_workers=8)
    data1 = data1.map(operations=image_ops2, input_columns="image", num_parallel_workers=8)

    data1 = data1.batch(batch_size=3, num_parallel_workers=8)
    data1 = data1.repeat(5)

    util_check_serialize_deserialize_file(data1, "complex1_dataset_pipeline", remove_json_files)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)

    if remove_json_files:
        delete_json_files("complex1_dataset_pipeline")


def test_serdes_fill(remove_json_files=True):
    """
    Feature: Serialize and Deserialize Support
    Description: Test serialize and deserialize on pipeline with Fill op
    Expectation: Serialized versus Deserialized+reserialized pipeline output verified
    """

    data_dir = "../data/dataset/testPK/data"
    data = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False, num_samples=11)

    label_fill_value = 3
    fill_op = c_transforms.Fill(label_fill_value)
    data = data.map(operations=fill_op, input_columns=["label"])

    for data1 in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(data1['label'], label_fill_value)

    util_check_serialize_deserialize_file(data, "fill_pipeline", remove_json_files)


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
    test_serdes_random_crop()
    test_serdes_random_rotation()
    test_serdes_pyvision()
    test_serdes_pyfunc_exception()
    test_serdes_pyfunc_exception2()
    test_serdes_inter_mixed_map()
    test_serdes_inter_mixed_enum_parms_map()
    test_serdes_intra_mixed_py2c_map()
    test_serdes_intra_mixed_c2py_map()
    test_serdes_uniform_augment()
    test_serdes_complex1_pipeline()
    test_serdes_fill()
