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
Testing dataset serialize and deserialize in DE
"""
import filecmp
import glob
import json
import numpy as np
import os
import pytest

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as c
import mindspore.dataset.transforms.vision.c_transforms as vision
from mindspore import log as logger
from mindspore.dataset.transforms.vision import Inter


def test_imagefolder(remove_json_files=True):
    """
    Test simulating resnet50 dataset pipeline.
    """
    data_dir = "../data/dataset/testPK/data"
    ds.config.set_seed(1)

    # define data augmentation parameters
    rescale = 1.0 / 255.0
    shift = 0.0
    resize_height, resize_width = 224, 224
    weights = [1.0, 0.1, 0.02, 0.3, 0.4, 0.05, 1.2, 0.13, 0.14, 0.015, 0.16, 1.1]

    # Constructing DE pipeline
    sampler = ds.WeightedRandomSampler(weights, 11)
    data1 = ds.ImageFolderDatasetV2(data_dir, sampler=sampler)
    data1 = data1.repeat(1)
    data1 = data1.map(input_columns=["image"], operations=[vision.Decode(True)])
    rescale_op = vision.Rescale(rescale, shift)

    resize_op = vision.Resize((resize_height, resize_width), Inter.LINEAR)
    data1 = data1.map(input_columns=["image"], operations=[rescale_op, resize_op])
    data1 = data1.batch(2)

    # Serialize the dataset pre-processing pipeline.
    # data1 should still work after saving.
    ds.serialize(data1, "imagenet_dataset_pipeline.json")
    ds1_dict = ds.serialize(data1)
    assert validate_jsonfile("imagenet_dataset_pipeline.json") is True

    # Print the serialized pipeline to stdout
    ds.show(data1)

    # Deserialize the serialized json file
    data2 = ds.deserialize(json_filepath="imagenet_dataset_pipeline.json")

    # Serialize the pipeline we just deserialized.
    # The content of the json file should be the same to the previous serialize.
    ds.serialize(data2, "imagenet_dataset_pipeline_1.json")
    assert validate_jsonfile("imagenet_dataset_pipeline_1.json") is True
    assert filecmp.cmp('imagenet_dataset_pipeline.json', 'imagenet_dataset_pipeline_1.json')

    # Deserialize the latest json file again
    data3 = ds.deserialize(json_filepath="imagenet_dataset_pipeline_1.json")
    data4 = ds.deserialize(input_dict=ds1_dict)
    num_samples = 0
    # Iterate and compare the data in the original pipeline (data1) against the deserialized pipeline (data2)
    for item1, item2, item3, item4 in zip(data1.create_dict_iterator(), data2.create_dict_iterator(),
                                          data3.create_dict_iterator(), data4.create_dict_iterator()):
        assert np.array_equal(item1['image'], item2['image'])
        assert np.array_equal(item1['image'], item3['image'])
        assert np.array_equal(item1['label'], item2['label'])
        assert np.array_equal(item1['label'], item3['label'])
        assert np.array_equal(item3['image'], item4['image'])
        assert np.array_equal(item3['label'], item4['label'])
        num_samples += 1

    logger.info("Number of data in data1: {}".format(num_samples))
    assert num_samples == 6

    # Remove the generated json file
    if remove_json_files:
        delete_json_files()


def test_mnist_dataset(remove_json_files=True):
    data_dir = "../data/dataset/testMnistData"
    ds.config.set_seed(1)

    data1 = ds.MnistDataset(data_dir, 100)
    one_hot_encode = c.OneHot(10)  # num_classes is input argument
    data1 = data1.map(input_columns="label", operations=one_hot_encode)

    # batch_size is input argument
    data1 = data1.batch(batch_size=10, drop_remainder=True)

    ds.serialize(data1, "mnist_dataset_pipeline.json")
    assert validate_jsonfile("mnist_dataset_pipeline.json") is True

    data2 = ds.deserialize(json_filepath="mnist_dataset_pipeline.json")
    ds.serialize(data2, "mnist_dataset_pipeline_1.json")
    assert validate_jsonfile("mnist_dataset_pipeline_1.json") is True
    assert filecmp.cmp('mnist_dataset_pipeline.json', 'mnist_dataset_pipeline_1.json')

    data3 = ds.deserialize(json_filepath="mnist_dataset_pipeline_1.json")

    num = 0
    for data1, data2, data3 in zip(data1.create_dict_iterator(), data2.create_dict_iterator(),
                                   data3.create_dict_iterator()):
        assert np.array_equal(data1['image'], data2['image'])
        assert np.array_equal(data1['image'], data3['image'])
        assert np.array_equal(data1['label'], data2['label'])
        assert np.array_equal(data1['label'], data3['label'])
        num += 1

    logger.info("mnist total num samples is {}".format(str(num)))
    assert num == 10

    if remove_json_files:
        delete_json_files()


def test_zip_dataset(remove_json_files=True):
    files = ["../data/dataset/testTFTestAllTypes/test.data"]
    schema_file = "../data/dataset/testTFTestAllTypes/datasetSchema.json"
    ds.config.set_seed(1)

    ds0 = ds.TFRecordDataset(files, schema=schema_file, shuffle=ds.Shuffle.GLOBAL)
    data1 = ds.TFRecordDataset(files, schema=schema_file, shuffle=ds.Shuffle.GLOBAL)
    data2 = ds.TFRecordDataset(files, schema=schema_file, shuffle=ds.Shuffle.FILES)
    data2 = data2.shuffle(10000)
    data2 = data2.rename(input_columns=["col_sint16", "col_sint32", "col_sint64", "col_float",
                                        "col_1d", "col_2d", "col_3d", "col_binary"],
                         output_columns=["column_sint16", "column_sint32", "column_sint64", "column_float",
                                         "column_1d", "column_2d", "column_3d", "column_binary"])
    data3 = ds.zip((data1, data2))
    ds.serialize(data3, "zip_dataset_pipeline.json")
    assert validate_jsonfile("zip_dataset_pipeline.json") is True
    assert validate_jsonfile("zip_dataset_pipeline_typo.json") is False

    data4 = ds.deserialize(json_filepath="zip_dataset_pipeline.json")
    ds.serialize(data4, "zip_dataset_pipeline_1.json")
    assert validate_jsonfile("zip_dataset_pipeline_1.json") is True
    assert filecmp.cmp('zip_dataset_pipeline.json', 'zip_dataset_pipeline_1.json')

    rows = 0
    for d0, d3, d4 in zip(ds0, data3, data4):
        num_cols = len(d0)
        offset = 0
        for t1 in d0:
            assert np.array_equal(t1, d3[offset])
            assert np.array_equal(t1, d3[offset + num_cols])
            assert np.array_equal(t1, d4[offset])
            assert np.array_equal(t1, d4[offset + num_cols])
            offset += 1
        rows += 1
    assert rows == 12

    if remove_json_files:
        delete_json_files()


def test_random_crop():
    logger.info("test_random_crop")
    DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
    SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"])
    decode_op = vision.Decode()
    random_crop_op = vision.RandomCrop([512, 512], [200, 200, 200, 200])
    data1 = data1.map(input_columns="image", operations=decode_op)
    data1 = data1.map(input_columns="image", operations=random_crop_op)

    # Serializing into python dictionary
    ds1_dict = ds.serialize(data1)
    # Serializing into json object
    ds1_json = json.dumps(ds1_dict, indent=2)

    # Reconstruct dataset pipeline from its serialized form
    data1_1 = ds.deserialize(input_dict=ds1_dict)

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"])
    data2 = data2.map(input_columns="image", operations=decode_op)

    for item1, item1_1, item2 in zip(data1.create_dict_iterator(), data1_1.create_dict_iterator(),
                                     data2.create_dict_iterator()):
        assert np.array_equal(item1['image'], item1_1['image'])
        image2 = item2["image"]


def validate_jsonfile(filepath):
    try:
        file_exist = os.path.exists(filepath)
        with open(filepath, 'r') as jfile:
            loaded_json = json.load(jfile)
    except IOError:
        return False
    return file_exist and isinstance(loaded_json, dict)


def delete_json_files():
    file_list = glob.glob('*.json')
    for f in file_list:
        try:
            os.remove(f)
        except IOError:
            logger.info("Error while deleting: {}".format(f))


# Test save load minddataset
from test_minddataset_sampler import add_and_remove_cv_file, get_data, CV_DIR_NAME, CV_FILE_NAME, FILES_NUM, \
    FileWriter, Inter


def test_minddataset(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    indices = [1, 2, 3, 5, 7]
    sampler = ds.SubsetRandomSampler(indices)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                              sampler=sampler)

    # Serializing into python dictionary
    ds1_dict = ds.serialize(data_set)
    # Serializing into json object
    ds1_json = json.dumps(ds1_dict, sort_keys=True)

    # Reconstruct dataset pipeline from its serialized form
    data_set = ds.deserialize(input_dict=ds1_dict)
    ds2_dict = ds.serialize(data_set)
    # Serializing into json object
    ds2_json = json.dumps(ds2_dict, sort_keys=True)

    assert ds1_json == ds2_json

    data = get_data(CV_DIR_NAME)
    assert data_set.get_dataset_size() == 5
    num_iter = 0
    for _ in data_set.create_dict_iterator():
        num_iter += 1
    assert num_iter == 5


if __name__ == '__main__':
    test_imagefolder()
    test_zip_dataset()
    test_mnist_dataset()
    test_random_crop()
