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
import mindspore.dataset.vision as vision

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


def test_mnist_dataset_size():
    """
    Feature: get_dataset_size
    Description: Test get_dataset_size on MnistDataset
    Expectation: The dataset is processed as expected
    """
    ds_total = ds.MnistDataset(MNIST_DATA_DIR)
    assert ds_total.get_dataset_size() == 10000
    assert len(ds_total) == 10000

    # test get dataset_size with the usage arg
    test_dataset = ds.MnistDataset(MNIST_DATA_DIR, usage="test")
    train_dataset = ds.MnistDataset(MNIST_DATA_DIR, usage="train")
    all_dataset = ds.MnistDataset(MNIST_DATA_DIR, usage="all")

    test_size = test_dataset.get_dataset_size()
    assert test_size == 10000
    assert len(test_dataset) == 10000

    train_size = train_dataset.get_dataset_size()
    assert train_size == 0
    assert len(train_dataset) == train_size

    all_size = all_dataset.get_dataset_size()
    assert all_size == 10000
    assert len(all_dataset) == 10000

    ds_shard_1_0 = ds.MnistDataset(MNIST_DATA_DIR, num_shards=1, shard_id=0)
    assert ds_shard_1_0.get_dataset_size() == 10000
    assert len(ds_shard_1_0) == 10000

    ds_shard_2_0 = ds.MnistDataset(MNIST_DATA_DIR, num_shards=2, shard_id=0)
    assert ds_shard_2_0.get_dataset_size() == 5000
    assert len(ds_shard_2_0) == 5000

    ds_shard_3_0 = ds.MnistDataset(MNIST_DATA_DIR, num_shards=3, shard_id=0)
    assert ds_shard_3_0.get_dataset_size() == 3334
    assert len(ds_shard_3_0) == 3334


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


def test_cifar10_dataset_size():
    """
    Feature: get_dataset_size
    Description: Test get_dataset_size on Cifar10Dataset
    Expectation: The dataset is processed as expected
    """
    ds_total = ds.Cifar10Dataset(CIFAR10_DATA_DIR)
    assert ds_total.get_dataset_size() == 10000
    assert len(ds_total) == 10000

    # test get_dataset_size with usage flag
    train_cifar10dataset = ds.Cifar10Dataset(CIFAR10_DATA_DIR, usage="train")
    train_cifar100dataset = ds.Cifar100Dataset(CIFAR100_DATA_DIR, usage="train")
    all_cifar10dataset = ds.Cifar10Dataset(CIFAR10_DATA_DIR, usage="all")

    train_size = train_cifar100dataset.get_dataset_size()
    assert train_size == 0
    assert len(train_cifar100dataset) == train_size

    train_size = train_cifar10dataset.get_dataset_size()
    assert train_size == 10000
    assert len(train_cifar10dataset) == 10000

    all_size = all_cifar10dataset.get_dataset_size()
    assert all_size == 10000
    assert len(all_cifar10dataset) == 10000

    ds_shard_1_0 = ds.Cifar10Dataset(CIFAR10_DATA_DIR, num_shards=1, shard_id=0)
    assert ds_shard_1_0.get_dataset_size() == 10000
    assert len(ds_shard_1_0) == 10000

    ds_shard_2_0 = ds.Cifar10Dataset(CIFAR10_DATA_DIR, num_shards=2, shard_id=0)
    assert ds_shard_2_0.get_dataset_size() == 5000
    assert len(ds_shard_2_0) == 5000

    ds_shard_3_0 = ds.Cifar10Dataset(CIFAR10_DATA_DIR, num_shards=3, shard_id=0)
    assert ds_shard_3_0.get_dataset_size() == 3334
    assert len(ds_shard_3_0) == 3334

    ds_shard_7_0 = ds.Cifar10Dataset(CIFAR10_DATA_DIR, num_shards=7, shard_id=0)
    assert ds_shard_7_0.get_dataset_size() == 1429
    assert len(ds_shard_7_0) == 1429


def test_cifar100_dataset_size():
    """
    Feature: get_dataset_size
    Description: Test get_dataset_size on Cifar100Dataset
    Expectation: The dataset is processed as expected
    """
    ds_total = ds.Cifar100Dataset(CIFAR100_DATA_DIR)
    assert ds_total.get_dataset_size() == 10000
    assert len(ds_total) == 10000

    # test get_dataset_size with usage flag
    test_cifar100dataset = ds.Cifar100Dataset(CIFAR100_DATA_DIR, usage="test")
    all_cifar100dataset = ds.Cifar100Dataset(CIFAR100_DATA_DIR, usage="all")

    test_size = test_cifar100dataset.get_dataset_size()
    assert test_size == 10000
    assert len(test_cifar100dataset) == 10000

    all_size = all_cifar100dataset.get_dataset_size()
    assert all_size == 10000
    assert len(all_cifar100dataset) == 10000

    ds_shard_1_0 = ds.Cifar100Dataset(CIFAR100_DATA_DIR, num_shards=1, shard_id=0)
    assert ds_shard_1_0.get_dataset_size() == 10000
    assert len(ds_shard_1_0) == 10000

    ds_shard_2_0 = ds.Cifar100Dataset(CIFAR100_DATA_DIR, num_shards=2, shard_id=0)
    assert ds_shard_2_0.get_dataset_size() == 5000
    assert len(ds_shard_2_0) == 5000

    ds_shard_3_0 = ds.Cifar100Dataset(CIFAR100_DATA_DIR, num_shards=3, shard_id=0)
    assert ds_shard_3_0.get_dataset_size() == 3334
    assert len(ds_shard_3_0) == 3334


def test_voc_dataset_size():
    """
    Feature: get_dataset_size
    Description: Test get_dataset_size on VOCDataset
    Expectation: The dataset is processed as expected
    """
    dataset = ds.VOCDataset(VOC_DATA_DIR, task="Segmentation", usage="train", shuffle=False, decode=True)
    assert dataset.get_dataset_size() == 10
    assert len(dataset) == 10

    dataset_shard_2_0 = ds.VOCDataset(VOC_DATA_DIR, task="Segmentation", usage="train", shuffle=False, decode=True,
                                      num_shards=2, shard_id=0)
    assert dataset_shard_2_0.get_dataset_size() == 5
    assert len(dataset_shard_2_0) == 5


def test_coco_dataset_size():
    """
    Feature: get_dataset_size
    Description: Test get_dataset_size on CocoDataset
    Expectation: The dataset is processed as expected
    """
    dataset = ds.CocoDataset(COCO_DATA_DIR, annotation_file=ANNOTATION_FILE, task="Detection",
                             decode=True, shuffle=False)
    assert dataset.get_dataset_size() == 6
    assert len(dataset) == 6

    dataset_shard_2_0 = ds.CocoDataset(COCO_DATA_DIR, annotation_file=ANNOTATION_FILE, task="Detection", decode=True,
                                       shuffle=False, num_shards=2, shard_id=0)
    assert dataset_shard_2_0.get_dataset_size() == 3
    assert len(dataset_shard_2_0) == 3


def test_celeba_dataset_size():
    """
    Feature: get_dataset_size
    Description: Test get_dataset_size on CelebADataset
    Expectation: The dataset is processed as expected
    """
    dataset = ds.CelebADataset(CELEBA_DATA_DIR, shuffle=False, decode=True)
    assert dataset.get_dataset_size() == 4
    assert len(dataset) == 4

    dataset_shard_2_0 = ds.CelebADataset(CELEBA_DATA_DIR, shuffle=False, decode=True, num_shards=2, shard_id=0)
    assert dataset_shard_2_0.get_dataset_size() == 2
    assert len(dataset_shard_2_0) == 2


def test_clue_dataset_size():
    """
    Feature: get_dataset_size
    Description: Test get_dataset_size on CLUEDataset
    Expectation: The dataset is processed as expected
    """
    dataset = ds.CLUEDataset(CLUE_FILE, task='AFQMC', usage='train', shuffle=False)
    assert dataset.get_dataset_size() == 3
    assert len(dataset) == 3

    dataset_shard_2_0 = ds.CLUEDataset(CLUE_FILE, task='AFQMC', usage='train', shuffle=False, num_shards=2, shard_id=0)
    assert dataset_shard_2_0.get_dataset_size() == 2
    assert len(dataset_shard_2_0) == 2


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


def test_pipeline_get_dataset_size():
    """
    Feature: get_dataset_size
    Description: Test get_dataset_size for TFRecordDataset that is pipelined
    Expectation: The dataset is processed as expected
    """
    dataset = ds.TFRecordDataset(IMAGENET_TFFILE_DIR, SCHEMA_FILE, columns_list=["image"], shuffle=False)
    assert dataset.get_dataset_size() == 12
    assert len(dataset) == 12

    dataset = dataset.shuffle(buffer_size=3)
    assert dataset.get_dataset_size() == 12
    assert len(dataset) == 12

    decode_op = vision.Decode()
    resize_op = vision.RandomResize(10)

    dataset = dataset.map([decode_op, resize_op], input_columns=["image"])
    assert dataset.get_dataset_size() == 12
    assert len(dataset) == 12

    dataset = dataset.batch(batch_size=3)
    assert dataset.get_dataset_size() == 4
    assert len(dataset) == 4

    dataset = dataset.repeat(count=2)
    assert dataset.get_dataset_size() == 8
    assert len(dataset) == 8

    tf1 = ds.TFRecordDataset(IMAGENET_TFFILE_DIR, shuffle=True)
    tf2 = ds.TFRecordDataset(IMAGENET_TFFILE_DIR, shuffle=True)
    tf3 = tf2.concat(tf1)
    assert tf3.get_dataset_size() == 24
    assert len(tf3) == 24


def test_distributed_get_dataset_size():
    """
    Feature: get_dataset_size
    Description: Test get_dataset_size for MnistDataset with num_samples, num_shards, and shard_id
    Expectation: The dataset is processed as expected
    """
    # Test get dataset size when num_samples is less than num_per_shard (10000/4 = 2500)
    dataset1 = ds.MnistDataset(MNIST_DATA_DIR, num_samples=2000, num_shards=4, shard_id=0)
    assert dataset1.get_dataset_size() == 2000
    assert len(dataset1) == 2000

    count1 = 0
    for _ in dataset1.create_dict_iterator(num_epochs=1):
        count1 += 1
    assert count1 == 2000

    # Test get dataset size when num_samples is more than num_per_shard (10000/4 = 2500)
    dataset2 = ds.MnistDataset(MNIST_DATA_DIR, num_samples=3000, num_shards=4, shard_id=0)
    assert dataset2.get_dataset_size() == 2500
    assert len(dataset2) == 2500

    count2 = 0
    for _ in dataset2.create_dict_iterator(num_epochs=1):
        count2 += 1
    assert count2 == 2500


if __name__ == '__main__':
    test_imagenet_rawdata_dataset_size()
    test_imagenet_tf_file_dataset_size()
    test_mnist_dataset_size()
    test_mind_dataset_size()
    test_manifest_dataset_size()
    test_cifar10_dataset_size()
    test_cifar100_dataset_size()
    test_voc_dataset_size()
    test_coco_dataset_size()
    test_celeba_dataset_size()
    test_clue_dataset_size()
    test_csv_dataset_size()
    test_text_file_dataset_size()
    test_padded_dataset_size()
    test_pipeline_get_dataset_size()
    test_distributed_get_dataset_size()
