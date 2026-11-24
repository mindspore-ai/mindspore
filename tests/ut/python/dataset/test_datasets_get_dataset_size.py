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

"""Dataset regression coverage for datasets get dataset size."""
import mindspore.dataset as ds
from tests.dataset_mock import mnist_source

IMAGENET_RAWDATA_DIR = "../data/dataset/testImageNetData2/train"
IMAGENET_TFFILE_DIR = ["../data/dataset/test_tf_file_3_images2/train-0000-of-0001.data",
                       "../data/dataset/test_tf_file_3_images2/train-0000-of-0002.data",
                       "../data/dataset/test_tf_file_3_images2/train-0000-of-0003.data",
                       "../data/dataset/test_tf_file_3_images2/train-0000-of-0004.data"]
MNIST_DATA_DIR = "../data/dataset/testMnistData"
MIND_CV_FILE_NAME = "../data/mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord"
SCHEMA_FILE = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
MANIFEST_DATA_FILE = "../data/dataset/testManifestData/test.manifest"
CIFAR10_DATA_DIR = "../data/dataset/testCifar10Data"
CIFAR100_DATA_DIR = "../data/dataset/testCifar100Data"
VOC_DATA_DIR = "../data/dataset/testVOC2012"
COCO_DATA_DIR = "../data/dataset/testCOCO/train/"
ANNOTATION_FILE = "../data/dataset/testCOCO/annotations/train.json"
CELEBA_DATA_DIR = "../data/dataset/testCelebAData/"
CLUE_FILE = '../data/dataset/testCLUE/afqmc/train.json'
CSV_FILE = '../data/dataset/testCSV/1.csv'
TEXT_DATA_FILE = "../data/dataset/testTextFileDataset/1.txt"


def test_imagenet_rawdata_dataset_size():
    """
    Feature: get_dataset_size
    Description: Test get_dataset_size on ImageFolderDataset
    Expectation: The dataset is processed as expected
    """
    ds_total = ds.ImageFolderDataset(IMAGENET_RAWDATA_DIR)
    assert ds_total.get_dataset_size() == 6
    assert len(ds_total) == 6

    ds_shard_1_0 = ds.ImageFolderDataset(IMAGENET_RAWDATA_DIR, num_shards=1, shard_id=0)
    assert ds_shard_1_0.get_dataset_size() == 6
    assert len(ds_shard_1_0) == 6

    ds_shard_2_0 = ds.ImageFolderDataset(IMAGENET_RAWDATA_DIR, num_shards=2, shard_id=0)
    assert ds_shard_2_0.get_dataset_size() == 3
    assert len(ds_shard_2_0) == 3

    ds_shard_3_0 = ds.ImageFolderDataset(IMAGENET_RAWDATA_DIR, num_shards=3, shard_id=0)
    assert ds_shard_3_0.get_dataset_size() == 2
    assert len(ds_shard_3_0) == 2


def test_imagenet_tf_file_dataset_size():
    """
    Feature: get_dataset_size
    Description: Test get_dataset_size on TFRecordDataset
    Expectation: The dataset is processed as expected
    """
    ds_total = ds.TFRecordDataset(IMAGENET_TFFILE_DIR)
    assert ds_total.get_dataset_size() == 12
    assert len(ds_total) == 12

    ds_shard_1_0 = ds.TFRecordDataset(IMAGENET_TFFILE_DIR, num_shards=1, shard_id=0, shard_equal_rows=True)
    assert ds_shard_1_0.get_dataset_size() == 12
    assert len(ds_shard_1_0) == 12

    ds_shard_2_0 = ds.TFRecordDataset(IMAGENET_TFFILE_DIR, num_shards=2, shard_id=0, shard_equal_rows=True)
    assert ds_shard_2_0.get_dataset_size() == 6
    assert len(ds_shard_2_0) == 6

    ds_shard_3_0 = ds.TFRecordDataset(IMAGENET_TFFILE_DIR, num_shards=3, shard_id=0, shard_equal_rows=True)
    assert ds_shard_3_0.get_dataset_size() == 4
    assert len(ds_shard_3_0) == 4

    count = 0
    for _ in ds_shard_3_0.create_dict_iterator(num_epochs=1):
        count += 1
    assert ds_shard_3_0.get_dataset_size() == count
    assert len(ds_shard_3_0) == count

    # shard_equal_rows is set to False therefore, get_dataset_size must return count
    ds_shard_4_0 = ds.TFRecordDataset(IMAGENET_TFFILE_DIR, num_shards=4, shard_id=0)
    count = 0
    for _ in ds_shard_4_0.create_dict_iterator(num_epochs=1):
        count += 1
    assert ds_shard_4_0.get_dataset_size() == count
    assert len(ds_shard_4_0) == count


def test_mind_dataset_size():
    """
    Feature: get_dataset_size
    Description: Test get_dataset_size on MindDataset
    Expectation: The dataset is processed as expected
    """
    dataset = ds.MindDataset(MIND_CV_FILE_NAME + "0")
    assert dataset.get_dataset_size() == 20
    assert len(dataset) == 20

    dataset_shard_2_0 = ds.MindDataset(MIND_CV_FILE_NAME + "0", num_shards=2, shard_id=0)
    assert dataset_shard_2_0.get_dataset_size() == 10
    assert len(dataset_shard_2_0) == 10


def test_manifest_dataset_size():
    """
    Feature: get_dataset_size
    Description: Test get_dataset_size on ManifestDataset
    Expectation: The dataset is processed as expected
    """
    ds_total = ds.ManifestDataset(MANIFEST_DATA_FILE)
    assert ds_total.get_dataset_size() == 4
    assert len(ds_total) == 4

    ds_shard_1_0 = ds.ManifestDataset(MANIFEST_DATA_FILE, num_shards=1, shard_id=0)
    assert ds_shard_1_0.get_dataset_size() == 4
    assert len(ds_shard_1_0) == 4

    ds_shard_2_0 = ds.ManifestDataset(MANIFEST_DATA_FILE, num_shards=2, shard_id=0)
    assert ds_shard_2_0.get_dataset_size() == 2
    assert len(ds_shard_2_0) == 2

    ds_shard_3_0 = ds.ManifestDataset(MANIFEST_DATA_FILE, num_shards=3, shard_id=0)
    assert ds_shard_3_0.get_dataset_size() == 2
    assert len(ds_shard_3_0) == 2


def test_csv_dataset_size():
    """
    Feature: get_dataset_size
    Description: Test get_dataset_size on CSVDataset
    Expectation: The dataset is processed as expected
    """
    dataset = ds.CSVDataset(CSV_FILE, column_defaults=["0", 0, 0.0, "0"], column_names=['1', '2', '3', '4'],
                            shuffle=False)
    assert dataset.get_dataset_size() == 3
    assert len(dataset) == 3

    dataset_shard_2_0 = ds.CSVDataset(CSV_FILE, column_defaults=["0", 0, 0.0, "0"], column_names=['1', '2', '3', '4'],
                                      shuffle=False, num_shards=2, shard_id=0)
    assert dataset_shard_2_0.get_dataset_size() == 2
    assert len(dataset_shard_2_0) == 2


def test_text_file_dataset_size():
    """
    Feature: get_dataset_size
    Description: Test get_dataset_size on TextFileDataset
    Expectation: The dataset is processed as expected
    """
    dataset = ds.TextFileDataset(TEXT_DATA_FILE)
    assert dataset.get_dataset_size() == 3
    assert len(dataset) == 3

    dataset_shard_2_0 = ds.TextFileDataset(TEXT_DATA_FILE, num_shards=2, shard_id=0)
    assert dataset_shard_2_0.get_dataset_size() == 2
    assert len(dataset_shard_2_0) == 2


def test_padded_dataset_size():
    """
    Feature: get_dataset_size
    Description: Test get_dataset_size on PaddedDataset
    Expectation: The dataset is processed as expected
    """
    dataset = ds.PaddedDataset([{"data": [1, 2, 3]}, {"data": [1, 0, 1]}])
    assert dataset.get_dataset_size() == 2
    assert len(dataset) == 2




def test_distributed_get_dataset_size():
    """
    Feature: get_dataset_size
    Description: Test get_dataset_size for MnistDataset with num_samples, num_shards, and shard_id
    Expectation: The dataset is processed as expected
    """
    # Test get dataset size when num_samples is less than num_per_shard (10000/4 = 2500)
    dataset1 = mnist_source(MNIST_DATA_DIR, num_samples=2000, num_shards=4, shard_id=0)
    assert dataset1.get_dataset_size() == 2000
    assert len(dataset1) == 2000

    count1 = 0
    for _ in dataset1.create_dict_iterator(num_epochs=1):
        count1 += 1
    assert count1 == 2000

    # Test get dataset size when num_samples is more than num_per_shard (10000/4 = 2500)
    dataset2 = mnist_source(MNIST_DATA_DIR, num_samples=3000, num_shards=4, shard_id=0)
    assert dataset2.get_dataset_size() == 2500
    assert len(dataset2) == 2500

    count2 = 0
    for _ in dataset2.create_dict_iterator(num_epochs=1):
        count2 += 1
    assert count2 == 2500
