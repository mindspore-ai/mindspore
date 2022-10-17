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
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision as vision

CELEBA_DIR = "../data/dataset/testCelebAData"
CIFAR10_DIR = "../data/dataset/testCifar10Data"
CIFAR100_DIR = "../data/dataset/testCifar100Data"
CLUE_DIR = "../data/dataset/testCLUE/afqmc/train.json"
COCO_DIR = "../data/dataset/testCOCO/train"
COCO_ANNOTATION = "../data/dataset/testCOCO/annotations/train.json"
CSV_DIR = "../data/dataset/testCSV/1.csv"
IMAGE_FOLDER_DIR = "../data/dataset/testPK/data/"
MANIFEST_DIR = "../data/dataset/testManifestData/test.manifest"
MNIST_DIR = "../data/dataset/testMnistData"
TFRECORD_DIR = ["../data/dataset/testTFTestAllTypes/test.data"]
TFRECORD_SCHEMA = "../data/dataset/testTFTestAllTypes/datasetSchema.json"
VOC_DIR = "../data/dataset/testVOC2012"


def test_get_column_name_celeba():
    """
    Feature: get_col_names
    Description: Test get_col_names with CelebADataset
    Expectation: Output is equal to the expected output
    """
    data = ds.CelebADataset(CELEBA_DIR)
    assert data.get_col_names() == ["image", "attr"]


def test_get_column_name_cifar10():
    """
    Feature: get_col_names
    Description: Test get_col_names with Cifar10Dataset
    Expectation: Output is equal to the expected output
    """
    data = ds.Cifar10Dataset(CIFAR10_DIR)
    assert data.get_col_names() == ["image", "label"]


def test_get_column_name_cifar100():
    """
    Feature: get_col_names
    Description: Test get_col_names with Cifar100Dataset
    Expectation: Output is equal to the expected output
    """
    data = ds.Cifar100Dataset(CIFAR100_DIR)
    assert data.get_col_names() == ["image", "coarse_label", "fine_label"]


def test_get_column_name_clue():
    """
    Feature: get_col_names
    Description: Test get_col_names with CLUEDataset
    Expectation: Output is equal to the expected output
    """
    data = ds.CLUEDataset(CLUE_DIR, task="AFQMC", usage="train")
    assert data.get_col_names() == ["label", "sentence1", "sentence2"]


def test_get_column_name_coco():
    """
    Feature: get_col_names
    Description: Test get_col_names with CocoDataset
    Expectation: Output is equal to the expected output
    """
    data = ds.CocoDataset(COCO_DIR, annotation_file=COCO_ANNOTATION, task="Detection",
                          decode=True, shuffle=False)
    assert data.get_col_names() == ["image", "bbox", "category_id", "iscrowd"]


def test_get_column_name_csv():
    """
    Feature: get_col_names
    Description: Test get_col_names with CSVDataset
    Expectation: Output is equal to the expected output
    """
    data = ds.CSVDataset(CSV_DIR)
    assert data.get_col_names() == ["1", "2", "3", "4"]
    data = ds.CSVDataset(CSV_DIR, column_names=["col1", "col2", "col3", "col4"])
    assert data.get_col_names() == ["col1", "col2", "col3", "col4"]


def test_get_column_name_generator():
    """
    Feature: get_col_names
    Description: Test get_col_names with GeneratorDataset
    Expectation: Output is equal to the expected output
    """
    def generator():
        for i in range(64):
            yield (np.array([i]),)

    data = ds.GeneratorDataset(generator, ["data"])
    assert data.get_col_names() == ["data"]


def test_get_column_name_imagefolder():
    """
    Feature: get_col_names
    Description: Test get_col_names with ImageFolderDataset
    Expectation: Output is equal to the expected output
    """
    data = ds.ImageFolderDataset(IMAGE_FOLDER_DIR)
    assert data.get_col_names() == ["image", "label"]


def test_get_column_name_iterator():
    """
    Feature: get_col_names
    Description: Test get_col_names with iterator
    Expectation: Output is equal to the expected output
    """
    data = ds.Cifar10Dataset(CIFAR10_DIR)
    itr = data.create_tuple_iterator(num_epochs=1)
    assert itr.get_col_names() == ["image", "label"]
    itr = data.create_dict_iterator(num_epochs=1)
    assert itr.get_col_names() == ["image", "label"]


def test_get_column_name_manifest():
    """
    Feature: get_col_names
    Description: Test get_col_names with ManifestDataset
    Expectation: Output is equal to the expected output
    """
    data = ds.ManifestDataset(MANIFEST_DIR)
    assert data.get_col_names() == ["image", "label"]


def test_get_column_name_map():
    """
    Feature: get_col_names
    Description: Test get_col_names after a map operation
    Expectation: Output is equal to the expected output
    """
    data = ds.Cifar10Dataset(CIFAR10_DIR)
    center_crop_op = vision.CenterCrop(10)
    data = data.map(operations=center_crop_op, input_columns=["image"])
    assert data.get_col_names() == ["image", "label"]
    data = ds.Cifar10Dataset(CIFAR10_DIR)
    data = data.map(operations=center_crop_op, input_columns=["image"], output_columns=["image"])
    assert data.get_col_names() == ["image", "label"]
    data = ds.Cifar10Dataset(CIFAR10_DIR)
    data = data.map(operations=center_crop_op, input_columns=["image"], output_columns=["col1"])
    assert data.get_col_names() == ["col1", "label"]
    data = ds.Cifar10Dataset(CIFAR10_DIR)
    data = data.map(operations=center_crop_op, input_columns=["image"], output_columns=["col1", "col2"],
                    column_order=["col2", "col1"])
    assert data.get_col_names() == ["col2", "col1"]


def test_get_column_name_mnist():
    """
    Feature: get_col_names
    Description: Test get_col_names with MnistDataset
    Expectation: Output is equal to the expected output
    """
    data = ds.MnistDataset(MNIST_DIR)
    assert data.get_col_names() == ["image", "label"]


def test_get_column_name_numpy_slices():
    """
    Feature: get_col_names
    Description: Test get_col_names with NumpySlicesDataset
    Expectation: Output is equal to the expected output
    """
    np_data = {"a": [1, 2], "b": [3, 4]}
    data = ds.NumpySlicesDataset(np_data, shuffle=False)
    assert data.get_col_names() == ["a", "b"]
    data = ds.NumpySlicesDataset([1, 2, 3], shuffle=False)
    assert data.get_col_names() == ["column_0"]


def test_get_column_name_tfrecord():
    """
    Feature: get_col_names
    Description: Test get_col_names with TFRecordDataset
    Expectation: Output is equal to the expected output
    """
    data = ds.TFRecordDataset(TFRECORD_DIR, TFRECORD_SCHEMA)
    assert data.get_col_names() == ["col_1d", "col_2d", "col_3d", "col_binary", "col_float", "col_sint16", "col_sint32",
                                    "col_sint64"]
    data = ds.TFRecordDataset(TFRECORD_DIR, TFRECORD_SCHEMA,
                              columns_list=["col_sint16", "col_sint64", "col_2d", "col_binary"])
    assert data.get_col_names() == ["col_sint16", "col_sint64", "col_2d", "col_binary"]

    data = ds.TFRecordDataset(TFRECORD_DIR)
    assert data.get_col_names() == ["col_1d", "col_2d", "col_3d", "col_binary", "col_float", "col_sint16", "col_sint32",
                                    "col_sint64", "col_sint8"]
    s = ds.Schema()
    s.add_column("line", "string", [])
    s.add_column("words", "string", [-1])
    s.add_column("chinese", "string", [])

    data = ds.TFRecordDataset("../data/dataset/testTextTFRecord/text.tfrecord", shuffle=False, schema=s)
    assert data.get_col_names() == ["line", "words", "chinese"]


def test_get_column_name_device_que():
    """
    Feature: get_col_names
    Description: Test get_col_names after device_que operation
    Expectation: Output is equal to the expected output
    """
    data = ds.Cifar10Dataset(CIFAR10_DIR)
    data = data.device_que()
    assert data.get_col_names() == ["image", "label"]


def test_get_column_name_voc():
    """
    Feature: get_col_names
    Description: Test get_col_names with VOCDataset
    Expectation: Output is equal to the expected output
    """
    data = ds.VOCDataset(VOC_DIR, task="Segmentation", usage="train", decode=True, shuffle=False)
    assert data.get_col_names() == ["image", "target"]
    data = ds.VOCDataset(VOC_DIR, task="Segmentation", usage="train", decode=True, shuffle=False, extra_metadata=True)
    assert data.get_col_names() == ["image", "target", "_meta-filename"]


def test_get_column_name_project():
    """
    Feature: get_col_names
    Description: Test get_col_names after project operation
    Expectation: Output is equal to the expected output
    """
    data = ds.Cifar10Dataset(CIFAR10_DIR)
    assert data.get_col_names() == ["image", "label"]
    data = data.project(columns=["image"])
    assert data.get_col_names() == ["image"]


def test_get_column_name_rename():
    """
    Feature: get_col_names
    Description: Test get_col_names with after a rename operation
    Expectation: Output is equal to the expected output
    """
    data = ds.Cifar10Dataset(CIFAR10_DIR)
    assert data.get_col_names() == ["image", "label"]
    data = data.rename(["image", "label"], ["test1", "test2"])
    assert data.get_col_names() == ["test1", "test2"]


def test_get_column_name_zip():
    """
    Feature: get_col_names
    Description: Test get_col_names after zip operation
    Expectation: Output is equal to the expected output
    """
    data1 = ds.Cifar10Dataset(CIFAR10_DIR)
    assert data1.get_col_names() == ["image", "label"]
    data2 = ds.CSVDataset(CSV_DIR)
    assert data2.get_col_names() == ["1", "2", "3", "4"]
    data = ds.zip((data1, data2))
    assert data.get_col_names() == ["image", "label", "1", "2", "3", "4"]


if __name__ == "__main__":
    test_get_column_name_celeba()
    test_get_column_name_cifar10()
    test_get_column_name_cifar100()
    test_get_column_name_clue()
    test_get_column_name_coco()
    test_get_column_name_csv()
    test_get_column_name_generator()
    test_get_column_name_imagefolder()
    test_get_column_name_iterator()
    test_get_column_name_manifest()
    test_get_column_name_map()
    test_get_column_name_mnist()
    test_get_column_name_numpy_slices()
    test_get_column_name_tfrecord()
    test_get_column_name_device_que()
    test_get_column_name_voc()
    test_get_column_name_project()
    test_get_column_name_rename()
    test_get_column_name_zip()
