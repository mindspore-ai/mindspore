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

import mindspore.dataset as ds

IMAGENET_RAWDATA_DIR = "../data/dataset/testImageNetData2/train"
IMAGENET_TFFILE_DIR = ["../data/dataset/test_tf_file_3_images2/train-0000-of-0001.data",
                       "../data/dataset/test_tf_file_3_images2/train-0000-of-0002.data",
                       "../data/dataset/test_tf_file_3_images2/train-0000-of-0003.data",
                       "../data/dataset/test_tf_file_3_images2/train-0000-of-0004.data"]
MNIST_DATA_DIR = "../data/dataset/testMnistData"
MANIFEST_DATA_FILE = "../data/dataset/testManifestData/test.manifest"
CIFAR10_DATA_DIR = "../data/dataset/testCifar10Data"
CIFAR100_DATA_DIR = "../data/dataset/testCifar100Data"


def test_imagenet_rawdata_dataset_size():
    ds_total = ds.ImageFolderDataset(IMAGENET_RAWDATA_DIR)
    assert ds_total.get_dataset_size() == 6

    ds_shard_1_0 = ds.ImageFolderDataset(IMAGENET_RAWDATA_DIR, num_shards=1, shard_id=0)
    assert ds_shard_1_0.get_dataset_size() == 6

    ds_shard_2_0 = ds.ImageFolderDataset(IMAGENET_RAWDATA_DIR, num_shards=2, shard_id=0)
    assert ds_shard_2_0.get_dataset_size() == 3

    ds_shard_3_0 = ds.ImageFolderDataset(IMAGENET_RAWDATA_DIR, num_shards=3, shard_id=0)
    assert ds_shard_3_0.get_dataset_size() == 2


def test_imagenet_tf_file_dataset_size():
    ds_total = ds.TFRecordDataset(IMAGENET_TFFILE_DIR)
    assert ds_total.get_dataset_size() == 12

    ds_shard_1_0 = ds.TFRecordDataset(IMAGENET_TFFILE_DIR, num_shards=1, shard_id=0)
    assert ds_shard_1_0.get_dataset_size() == 12

    ds_shard_2_0 = ds.TFRecordDataset(IMAGENET_TFFILE_DIR, num_shards=2, shard_id=0)
    assert ds_shard_2_0.get_dataset_size() == 6

    ds_shard_3_0 = ds.TFRecordDataset(IMAGENET_TFFILE_DIR, num_shards=3, shard_id=0)
    assert ds_shard_3_0.get_dataset_size() == 4


def test_mnist_dataset_size():
    ds_total = ds.MnistDataset(MNIST_DATA_DIR)
    assert ds_total.get_dataset_size() == 10000

    # test get dataset_size with the usage arg
    test_size = ds.MnistDataset(MNIST_DATA_DIR, usage="test").get_dataset_size()
    assert test_size == 10000
    train_size = ds.MnistDataset(MNIST_DATA_DIR, usage="train").get_dataset_size()
    assert train_size == 0
    all_size = ds.MnistDataset(MNIST_DATA_DIR, usage="all").get_dataset_size()
    assert all_size == 10000

    ds_shard_1_0 = ds.MnistDataset(MNIST_DATA_DIR, num_shards=1, shard_id=0)
    assert ds_shard_1_0.get_dataset_size() == 10000

    ds_shard_2_0 = ds.MnistDataset(MNIST_DATA_DIR, num_shards=2, shard_id=0)
    assert ds_shard_2_0.get_dataset_size() == 5000

    ds_shard_3_0 = ds.MnistDataset(MNIST_DATA_DIR, num_shards=3, shard_id=0)
    assert ds_shard_3_0.get_dataset_size() == 3334


def test_manifest_dataset_size():
    ds_total = ds.ManifestDataset(MANIFEST_DATA_FILE)
    assert ds_total.get_dataset_size() == 4

    ds_shard_1_0 = ds.ManifestDataset(MANIFEST_DATA_FILE, num_shards=1, shard_id=0)
    assert ds_shard_1_0.get_dataset_size() == 4

    ds_shard_2_0 = ds.ManifestDataset(MANIFEST_DATA_FILE, num_shards=2, shard_id=0)
    assert ds_shard_2_0.get_dataset_size() == 2

    ds_shard_3_0 = ds.ManifestDataset(MANIFEST_DATA_FILE, num_shards=3, shard_id=0)
    assert ds_shard_3_0.get_dataset_size() == 2


def test_cifar10_dataset_size():
    ds_total = ds.Cifar10Dataset(CIFAR10_DATA_DIR)
    assert ds_total.get_dataset_size() == 10000

    # test get_dataset_size with usage flag
    train_size = ds.Cifar10Dataset(CIFAR10_DATA_DIR, usage="train").get_dataset_size()
    assert train_size == 10000
    test_size = ds.Cifar10Dataset(CIFAR10_DATA_DIR, usage="test").get_dataset_size()
    assert test_size == 0
    all_size = ds.Cifar10Dataset(CIFAR10_DATA_DIR, usage="all").get_dataset_size()
    assert all_size == 10000

    ds_shard_1_0 = ds.Cifar10Dataset(CIFAR10_DATA_DIR, num_shards=1, shard_id=0)
    assert ds_shard_1_0.get_dataset_size() == 10000

    ds_shard_2_0 = ds.Cifar10Dataset(CIFAR10_DATA_DIR, num_shards=2, shard_id=0)
    assert ds_shard_2_0.get_dataset_size() == 5000

    ds_shard_3_0 = ds.Cifar10Dataset(CIFAR10_DATA_DIR, num_shards=3, shard_id=0)
    assert ds_shard_3_0.get_dataset_size() == 3334

    ds_shard_7_0 = ds.Cifar10Dataset(CIFAR10_DATA_DIR, num_shards=7, shard_id=0)
    assert ds_shard_7_0.get_dataset_size() == 1429


def test_cifar100_dataset_size():
    ds_total = ds.Cifar100Dataset(CIFAR100_DATA_DIR)
    assert ds_total.get_dataset_size() == 10000

    # test get_dataset_size with usage flag
    train_size = ds.Cifar100Dataset(CIFAR100_DATA_DIR, usage="train").get_dataset_size()
    assert train_size == 0
    test_size = ds.Cifar100Dataset(CIFAR100_DATA_DIR, usage="test").get_dataset_size()
    assert test_size == 10000
    all_size = ds.Cifar100Dataset(CIFAR100_DATA_DIR, usage="all").get_dataset_size()
    assert all_size == 10000

    ds_shard_1_0 = ds.Cifar100Dataset(CIFAR100_DATA_DIR, num_shards=1, shard_id=0)
    assert ds_shard_1_0.get_dataset_size() == 10000

    ds_shard_2_0 = ds.Cifar100Dataset(CIFAR100_DATA_DIR, num_shards=2, shard_id=0)
    assert ds_shard_2_0.get_dataset_size() == 5000

    ds_shard_3_0 = ds.Cifar100Dataset(CIFAR100_DATA_DIR, num_shards=3, shard_id=0)
    assert ds_shard_3_0.get_dataset_size() == 3334


if __name__ == '__main__':
    test_imagenet_rawdata_dataset_size()
    test_imagenet_tf_file_dataset_size()
    test_mnist_dataset_size()
    test_manifest_dataset_size()
    test_cifar10_dataset_size()
    test_cifar100_dataset_size()
