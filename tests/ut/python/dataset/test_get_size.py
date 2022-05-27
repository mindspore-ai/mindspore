# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
import mindspore.dataset as ds
from mindspore import log as logger

FILES = ["../data/dataset/testTFTestAllTypes/test.data"]
DATASET_ROOT = "../data/dataset/testTFTestAllTypes/"
SCHEMA_FILE = "../data/dataset/testTFTestAllTypes/datasetSchema.json"
GENERATE_GOLDEN = False


def test_case1():
    """
    Feature: get_dataset_size, get_batch_size, and get_repeat_count
    Description: Test with TFRecordDataset with schema after each multiple ops
    Expectation: Output is equal to the expected output
    """
    data = ds.TFRecordDataset(FILES, SCHEMA_FILE)
    assert data.get_dataset_size() == 12
    assert data.get_batch_size() == 1
    assert data.get_repeat_count() == 1
    data = data.shuffle(100)
    assert data.get_dataset_size() == 12
    assert data.get_batch_size() == 1
    assert data.get_repeat_count() == 1
    data = data.batch(2)
    assert data.get_dataset_size() == 6
    assert data.get_batch_size() == 2
    assert data.get_repeat_count() == 1
    data = data.rename("col_sint64", "new_column")
    assert data.get_dataset_size() == 6
    assert data.get_batch_size() == 2
    assert data.get_repeat_count() == 1
    data = data.repeat(10)
    assert data.get_dataset_size() == 60
    assert data.get_batch_size() == 2
    assert data.get_repeat_count() == 10
    data = data.project(["new_column"])
    assert data.get_dataset_size() == 60
    assert data.get_batch_size() == 2
    assert data.get_repeat_count() == 10

    data2 = ds.TFRecordDataset(FILES, SCHEMA_FILE).batch(2).repeat(10)

    data1 = data.zip(data2)
    assert data1.get_dataset_size() == 60


def test_case2():
    """
    Feature: get_dataset_size
    Description: Test with TFRecordDataset with num_samples after each multiple ops
    Expectation: Output is equal to the expected output
    """
    data = ds.TFRecordDataset(FILES, num_samples=6)
    assert data.get_dataset_size() == 6
    data = data.shuffle(100)
    assert data.get_dataset_size() == 6
    data = data.batch(2)
    assert data.get_dataset_size() == 3
    data = data.rename("col_sint64", "new_column")
    assert data.get_dataset_size() == 3
    data = data.repeat(10)
    assert data.get_dataset_size() == 30
    data = data.project(["new_column"])
    assert data.get_dataset_size() == 30

    data2 = ds.TFRecordDataset(FILES, num_samples=6).batch(2).repeat(10)

    data1 = data.zip(data2)
    assert data1.get_dataset_size() == 30


def test_case3():
    """
    Feature: get_dataset_size
    Description: Test with TFRecordDataset with schema and columns_list in zipped datasets
    Expectation: Output is equal to the expected output
    """
    data1 = ds.TFRecordDataset(FILES, SCHEMA_FILE, columns_list=["col_sint64"]).batch(2).repeat(10).rename(
        ["col_sint64"], ["a1"])
    data2 = ds.TFRecordDataset(FILES, SCHEMA_FILE, columns_list=["col_sint64"]).batch(2).repeat(5).rename(
        ["col_sint64"], ["a2"])
    data3 = ds.TFRecordDataset(FILES, SCHEMA_FILE, columns_list=["col_sint64"]).batch(2).rename(["col_sint64"], ["a3"])

    data4 = ds.zip((data1, data2, data3))

    assert data4.get_dataset_size() == 6


def test_case4():
    """
    Feature: get_dataset_size
    Description: Test with TFRecordDataset with schema and columns_list after multiple dataset operations
    Expectation: Output is equal to the expected output
    """
    data1 = ds.TFRecordDataset(FILES, SCHEMA_FILE, columns_list=["col_sint64"]).batch(2).repeat(10).rename(
        ["col_sint64"], ["a1"])
    data2 = ds.TFRecordDataset(FILES, columns_list=["col_sint64"]).rename(["col_sint64"], ["a2"])
    assert data2.get_dataset_size() == 12
    data2 = data2.batch(2)
    assert data2.get_dataset_size() == 6
    data2 = data2.shuffle(100)
    assert data2.get_dataset_size() == 6
    data2 = data2.repeat(3)
    assert data2.get_dataset_size() == 18

    data3 = ds.zip((data1, data2))

    assert data3.get_dataset_size() == 18


def test_case5():
    """
    Feature: get_dataset_size
    Description: Test with TFRecordDataset with drop_remainder option in batch op
    Expectation: Output is equal to the expected output
    """
    data = ds.TFRecordDataset(FILES, num_samples=10).batch(3, drop_remainder=True)
    assert data.get_dataset_size() == 3
    data = ds.TFRecordDataset(FILES, num_samples=10).batch(3, drop_remainder=False)
    assert data.get_dataset_size() == 4


def test_cifar():
    """
    Feature: get_dataset_size
    Description: Test with Cifar10Dataset and Cifar100Dataset with and without num_samples
    Expectation: Output is equal to the expected output
    """
    data = ds.Cifar10Dataset("../data/dataset/testCifar10Data")
    assert data.get_dataset_size() == 10000

    data = ds.Cifar10Dataset("../data/dataset/testCifar10Data", num_samples=10)
    assert data.get_dataset_size() == 10

    data = ds.Cifar10Dataset("../data/dataset/testCifar10Data", num_samples=90000)
    assert data.get_dataset_size() == 10000

    data = ds.Cifar100Dataset("../data/dataset/testCifar100Data")
    assert data.get_dataset_size() == 10000

    data = ds.Cifar100Dataset("../data/dataset/testCifar100Data", num_samples=10)
    assert data.get_dataset_size() == 10

    data = ds.Cifar100Dataset("../data/dataset/testCifar100Data", num_samples=20000)
    assert data.get_dataset_size() == 10000


def test_mnist():
    """
    Feature: get_dataset_size
    Description: Test with MnistDataset with and without num_samples
    Expectation: Output is equal to the expected output
    """
    data = ds.MnistDataset("../data/dataset/testMnistData")
    logger.info("dataset.size: {}".format(data.get_dataset_size()))
    assert data.get_dataset_size() == 10000

    data = ds.MnistDataset("../data/dataset/testMnistData", num_samples=10)
    assert data.get_dataset_size() == 10

    data = ds.MnistDataset("../data/dataset/testMnistData", num_samples=90000)
    assert data.get_dataset_size() == 10000


def test_manifest():
    """
    Feature: get_dataset_size and num_classes
    Description: Test with ManifestDataset before and after shuffle op
    Expectation: Output is equal to the expected output
    """
    data = ds.ManifestDataset("../data/dataset/testManifestData/test.manifest")
    assert data.get_dataset_size() == 4
    assert data.num_classes() == 3

    data = data.shuffle(100)
    assert data.num_classes() == 3


def test_imagefolder():
    """
    Feature: get_dataset_size and num_classes
    Description: Test with ImageFolderDataset with and without num_samples or with valid and invalid class_indexing
    Expectation: Output is equal to the expected output or error is raised as expected for invalid class_indexing name
    """
    data = ds.ImageFolderDataset("../data/dataset/testPK/data/")
    assert data.get_dataset_size() == 44
    assert data.num_classes() == 4
    data = data.shuffle(100)
    assert data.num_classes() == 4

    data = ds.ImageFolderDataset("../data/dataset/testPK/data/", num_samples=10)
    assert data.get_dataset_size() == 10
    assert data.num_classes() == 4

    data = ds.ImageFolderDataset("../data/dataset/testPK/data/", class_indexing={"class1": 1, "class2": 22})
    assert data.num_classes() == 2

    data = ds.ImageFolderDataset("../data/dataset/testPK/data/", class_indexing={"class1": 1, "wrong name": 22})
    err_msg = ""
    try:
        data.num_classes()
    except RuntimeError as e:
        err_msg = str(e)
    assert "wrong name doesn't exist" in err_msg


if __name__ == '__main__':
    test_manifest()
    test_case1()
    test_case2()
    test_case3()
    test_case4()
    test_case5()
    test_imagefolder()
