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


def test_imagefolder_shardings(print_res=False):
    """
    Feature: Sharding
    Description: Test ImageFolderDataset sharding
    Expectation: The dataset is processed as expected
    """
    image_folder_dir = "../data/dataset/testPK/data"

    def sharding_config(num_shards, shard_id, num_samples, shuffle, class_index, repeat_cnt=1):
        data1 = ds.ImageFolderDataset(image_folder_dir, num_samples=num_samples, num_shards=num_shards,
                                      shard_id=shard_id,
                                      shuffle=shuffle, class_indexing=class_index, decode=True)
        data1 = data1.repeat(repeat_cnt)
        res = []
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            res.append(item["label"].item())
        if print_res:
            logger.info("labels of dataset: {}".format(res))
        return res

    # total 44 rows in dataset
    assert (sharding_config(4, 0, 5, False, dict()) == [0, 0, 0, 1, 1])  # 5 rows
    assert (sharding_config(4, 0, 12, False, dict()) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3])  # 11 rows
    assert (sharding_config(4, 3, None, False, dict()) == [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])  # 11 rows
    assert (sharding_config(1, 0, 55, False, dict()) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])  # 44 rows
    assert (sharding_config(2, 0, 55, False, dict()) == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])  # 22 rows
    assert (sharding_config(2, 1, 55, False, dict()) == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3])  # 22 rows
    # total 22 in dataset rows because of class indexing which takes only 2 folders
    assert len(sharding_config(4, 0, None, True, {"class1": 111, "class2": 999})) == 6
    assert len(sharding_config(4, 2, 3, True, {"class1": 111, "class2": 999})) == 3
    # test with repeat
    assert (sharding_config(4, 0, 12, False, dict(), 3) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3] * 3)
    assert (sharding_config(4, 0, 5, False, dict(), 5) == [0, 0, 0, 1, 1] * 5)
    assert len(sharding_config(5, 1, None, True, {"class1": 111, "class2": 999}, 4)) == 20


def test_tfrecord_shardings1(print_res=False):
    """
    Feature: Sharding
    Description: Test TFRecordDataset sharding with num_parallel_workers=1
    Expectation: The dataset is processed as expected
    """

    # total 40 rows in dataset
    tf_files = ["../data/dataset/tf_file_dataset/test1.data", "../data/dataset/tf_file_dataset/test2.data",
                "../data/dataset/tf_file_dataset/test3.data", "../data/dataset/tf_file_dataset/test4.data"]

    def sharding_config(num_shards, shard_id, num_samples, repeat_cnt=1):
        data1 = ds.TFRecordDataset(tf_files, num_shards=num_shards, shard_id=shard_id, num_samples=num_samples,
                                   shuffle=ds.Shuffle.FILES, num_parallel_workers=1)
        data1 = data1.repeat(repeat_cnt)
        res = []
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            res.append(item["scalars"][0])
        if print_res:
            logger.info("scalars of dataset: {}".format(res))
        return res

    assert sharding_config(2, 0, None, 1) == [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]  # 20 rows
    assert sharding_config(2, 1, None, 1) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]  # 20 rows
    assert sharding_config(2, 0, 3, 1) == [11, 12, 13]  # 3 rows
    assert sharding_config(2, 1, 3, 1) == [1, 2, 3]  # 3 rows
    assert sharding_config(2, 0, 40, 1) == [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]  # 20 rows
    assert sharding_config(2, 1, 40, 1) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]  # 20 rows
    assert sharding_config(2, 0, 55, 1) == [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]  # 20 rows
    assert sharding_config(2, 1, 55, 1) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]  # 20 rows
    assert sharding_config(3, 0, 8, 1) == [11, 12, 13, 14, 15, 16, 17, 18]  # 8 rows
    assert sharding_config(3, 1, 8, 1) == [1, 2, 3, 4, 5, 6, 7, 8]  # 8 rows
    assert sharding_config(3, 2, 8, 1) == [21, 22, 23, 24, 25, 26, 27, 28]  # 8 rows
    assert sharding_config(4, 0, 2, 1) == [11, 12]  # 2 rows
    assert sharding_config(4, 1, 2, 1) == [1, 2]  # 2 rows
    assert sharding_config(4, 2, 2, 1) == [21, 22]  # 2 rows
    assert sharding_config(4, 3, 2, 1) == [31, 32]  # 2 rows
    assert sharding_config(3, 0, 4, 2) == [11, 12, 13, 14, 21, 22, 23, 24]  # 8 rows
    assert sharding_config(3, 1, 4, 2) == [1, 2, 3, 4, 11, 12, 13, 14]  # 8 rows
    assert sharding_config(3, 2, 4, 2) == [21, 22, 23, 24, 31, 32, 33, 34]  # 8 rows


def test_tfrecord_shardings4(print_res=False):
    """
    Feature: Sharding
    Description: Test TFRecordDataset sharding with num_parallel_workers=4
    Expectation: The dataset is processed as expected
    """

    # total 40 rows in dataset
    tf_files = ["../data/dataset/tf_file_dataset/test1.data", "../data/dataset/tf_file_dataset/test2.data",
                "../data/dataset/tf_file_dataset/test3.data", "../data/dataset/tf_file_dataset/test4.data"]

    def sharding_config(num_shards, shard_id, num_samples, repeat_cnt=1):
        data1 = ds.TFRecordDataset(tf_files, num_shards=num_shards, shard_id=shard_id, num_samples=num_samples,
                                   shuffle=ds.Shuffle.FILES, num_parallel_workers=4)
        data1 = data1.repeat(repeat_cnt)
        res = []
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            res.append(item["scalars"][0])
        if print_res:
            logger.info("scalars of dataset: {}".format(res))
        return res

    def check_result(result_list, expect_length, expect_set):
        assert len(result_list) == expect_length
        assert set(result_list) == expect_set

    check_result(sharding_config(2, 0, None, 1), 20,
                 {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30})
    check_result(sharding_config(2, 1, None, 1), 20,
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40})
    check_result(sharding_config(2, 0, 3, 1), 3, {11, 12, 21})
    check_result(sharding_config(2, 1, 3, 1), 3, {1, 2, 31})
    check_result(sharding_config(2, 0, 40, 1), 20,
                 {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30})
    check_result(sharding_config(2, 1, 40, 1), 20,
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40})
    check_result(sharding_config(2, 0, 55, 1), 20,
                 {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30})
    check_result(sharding_config(2, 1, 55, 1), 20,
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40})
    check_result(sharding_config(3, 0, 8, 1), 8, {32, 33, 34, 11, 12, 13, 14, 31})
    check_result(sharding_config(3, 1, 8, 1), 8, {1, 2, 3, 4, 5, 6, 7, 8})
    check_result(sharding_config(3, 2, 8, 1), 8, {21, 22, 23, 24, 25, 26, 27, 28})
    check_result(sharding_config(4, 0, 2, 1), 2, {11, 12})
    check_result(sharding_config(4, 1, 2, 1), 2, {1, 2})
    check_result(sharding_config(4, 2, 2, 1), 2, {21, 22})
    check_result(sharding_config(4, 3, 2, 1), 2, {31, 32})
    check_result(sharding_config(3, 0, 4, 2), 8, {32, 1, 2, 11, 12, 21, 22, 31})
    check_result(sharding_config(3, 1, 4, 2), 8, {1, 2, 3, 4, 11, 12, 13, 14})
    check_result(sharding_config(3, 2, 4, 2), 8, {32, 33, 34, 21, 22, 23, 24, 31})


def test_manifest_shardings(print_res=False):
    """
    Feature: Sharding
    Description: Test ManifestDataset sharding
    Expectation: The dataset is processed as expected
    """
    manifest_file = "../data/dataset/testManifestData/test5trainimgs.json"

    def sharding_config(num_shards, shard_id, num_samples, shuffle, repeat_cnt=1):
        data1 = ds.ManifestDataset(manifest_file, num_samples=num_samples, num_shards=num_shards, shard_id=shard_id,
                                   shuffle=shuffle, decode=True)
        data1 = data1.repeat(repeat_cnt)
        res = []
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            res.append(item["label"].item())
        if print_res:
            logger.info("labels of dataset: {}".format(res))
        return res

    # 5 train images in total
    sharding_config(2, 0, None, False)
    assert (sharding_config(2, 0, None, False) == [0, 1, 1])
    assert (sharding_config(2, 1, None, False) == [0, 0, 0])
    assert (sharding_config(2, 0, 2, False) == [0, 1])
    assert (sharding_config(2, 1, 2, False) == [0, 0])
    # with repeat
    assert (sharding_config(2, 1, None, False, 3) == [0, 0, 0] * 3)
    assert (sharding_config(2, 0, 2, False, 5) == [0, 1] * 5)


def test_voc_shardings(print_res=False):
    """
    Feature: Sharding
    Description: Test VOCDataset sharding
    Expectation: The dataset is processed as expected
    """
    voc_dir = "../data/dataset/testVOC2012"

    def sharding_config(num_shards, shard_id, num_samples, shuffle, repeat_cnt=1):
        sampler = ds.DistributedSampler(num_shards, shard_id, shuffle=shuffle, num_samples=num_samples)
        data1 = ds.VOCDataset(voc_dir, decode=True, sampler=sampler)
        data1 = data1.repeat(repeat_cnt)
        res = []
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            res.append(item["image"].shape[0])
        if print_res:
            logger.info("labels of dataset: {}".format(res))
        return res

    # 10 images in total, always decode to get the shape
    # first dim of all 10 images [2268,2268,2268,2268,642,607,561,596,612,2268]
    # 3 shard_workers, 0th worker will get 0-th, 3nd, 6th and 9th image
    assert (sharding_config(3, 0, None, False, 2) == [2268, 2268, 561, 2268] * 2)
    # 3 shard_workers, 1st worker will get 1-st, 4nd, 7th and 0th image, the last one goes back bc of rounding up
    assert (sharding_config(3, 1, 5, False, 3) == [2268, 642, 596, 2268] * 3)
    # 3 shard_workers, 2nd worker will get 2nd, 5th, 8th and 11th (which is 1st)
    # then takes the first 2 bc num_samples = 2
    assert (sharding_config(3, 2, 2, False, 4) == [2268, 607] * 4)
    # test that each epoch, each shard_worker returns a different sample
    assert len(sharding_config(2, 0, None, True, 1)) == 5
    assert len(set(sharding_config(11, 0, None, True, 10))) > 1


def test_cifar10_shardings(print_res=False):
    """
    Feature: Sharding
    Description: Test Cifar10Dataset sharding
    Expectation: The dataset is processed as expected
    """
    cifar10_dir = "../data/dataset/testCifar10Data"

    def sharding_config(num_shards, shard_id, num_samples, shuffle, repeat_cnt=1):
        data1 = ds.Cifar10Dataset(cifar10_dir, num_shards=num_shards, shard_id=shard_id, num_samples=num_samples,
                                  shuffle=shuffle)
        data1 = data1.repeat(repeat_cnt)
        res = []
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            res.append(item["label"].item())
        if print_res:
            logger.info("labels of dataset: {}".format(res))
        return res

    # 10000 rows in total. CIFAR reads everything in memory which would make each test case very slow
    # therefore, only 2 test cases for now.
    assert sharding_config(10000, 9999, 7, False, 1) == [9]
    assert sharding_config(10000, 0, 4, False, 3) == [0, 0, 0]


def test_cifar100_shardings(print_res=False):
    """
    Feature: Sharding
    Description: Test Cifar100Dataset sharding
    Expectation: The dataset is processed as expected
    """
    cifar100_dir = "../data/dataset/testCifar100Data"

    def sharding_config(num_shards, shard_id, num_samples, shuffle, repeat_cnt=1):
        data1 = ds.Cifar100Dataset(cifar100_dir, num_shards=num_shards, shard_id=shard_id, num_samples=num_samples,
                                   shuffle=shuffle)
        data1 = data1.repeat(repeat_cnt)
        res = []
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            res.append(item["coarse_label"].item())
        if print_res:
            logger.info("labels of dataset: {}".format(res))
        return res

    # 10000 rows in total in test.bin CIFAR100 file
    assert (sharding_config(1000, 999, 7, False, 2) == [1, 18, 10, 17, 5, 0, 15] * 2)
    assert (sharding_config(1000, 0, None, False) == [10, 16, 2, 11, 10, 17, 11, 14, 13, 3])


def test_mnist_shardings(print_res=False):
    """
    Feature: Sharding
    Description: Test MnistDataset sharding
    Expectation: The dataset is processed as expected
    """
    mnist_dir = "../data/dataset/testMnistData"

    def sharding_config(num_shards, shard_id, num_samples, shuffle, repeat_cnt=1):
        data1 = ds.MnistDataset(mnist_dir, num_shards=num_shards, shard_id=shard_id, num_samples=num_samples,
                                shuffle=shuffle)
        data1 = data1.repeat(repeat_cnt)
        res = []
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            res.append(item["label"].item())
        if print_res:
            logger.info("labels of dataset: {}".format(res))
        return res

    # 70K rows in total , divide across 10K hosts, each host has 7 images
    assert sharding_config(10000, 0, num_samples=5, shuffle=False, repeat_cnt=3) == [0, 0, 0]
    assert sharding_config(10000, 9999, num_samples=None, shuffle=False, repeat_cnt=1) == [9]


if __name__ == '__main__':
    test_imagefolder_shardings(True)
    test_tfrecord_shardings1(True)
    test_tfrecord_shardings4(True)
    test_manifest_shardings(True)
    test_voc_shardings(True)
    test_cifar10_shardings(True)
    test_cifar100_shardings(True)
    test_mnist_shardings(True)
