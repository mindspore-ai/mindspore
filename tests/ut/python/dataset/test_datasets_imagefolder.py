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
from mindspore import log as logger

DATA_DIR = "../data/dataset/testPK/data"


def test_imagefolder_basic():
    logger.info("Test Case basic")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    data1 = ds.ImageFolderDatasetV2(DATA_DIR)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 44


def test_imagefolder_numsamples():
    logger.info("Test Case numSamples")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    data1 = ds.ImageFolderDatasetV2(DATA_DIR, num_samples=10, num_parallel_workers=2)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 10

    random_sampler = ds.RandomSampler(num_samples=3, replacement=True)
    data1 = ds.ImageFolderDatasetV2(DATA_DIR, num_parallel_workers=2, sampler=random_sampler)

    num_iter = 0
    for item in data1.create_dict_iterator():
        num_iter += 1

    assert num_iter == 3

    random_sampler = ds.RandomSampler(num_samples=3, replacement=False)
    data1 = ds.ImageFolderDatasetV2(DATA_DIR, num_parallel_workers=2, sampler=random_sampler)

    num_iter = 0
    for item in data1.create_dict_iterator():
        num_iter += 1

    assert num_iter == 3


def test_imagefolder_numshards():
    logger.info("Test Case numShards")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    data1 = ds.ImageFolderDatasetV2(DATA_DIR, num_shards=4, shard_id=3)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 11


def test_imagefolder_shardid():
    logger.info("Test Case withShardID")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    data1 = ds.ImageFolderDatasetV2(DATA_DIR, num_shards=4, shard_id=1)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 11


def test_imagefolder_noshuffle():
    logger.info("Test Case noShuffle")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    data1 = ds.ImageFolderDatasetV2(DATA_DIR, shuffle=False)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 44


def test_imagefolder_extrashuffle():
    logger.info("Test Case extraShuffle")
    # define parameters
    repeat_count = 2

    # apply dataset operations
    data1 = ds.ImageFolderDatasetV2(DATA_DIR, shuffle=True)
    data1 = data1.shuffle(buffer_size=5)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 88


def test_imagefolder_classindex():
    logger.info("Test Case classIndex")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    class_index = {"class3": 333, "class1": 111}
    data1 = ds.ImageFolderDatasetV2(DATA_DIR, class_indexing=class_index, shuffle=False)
    data1 = data1.repeat(repeat_count)

    golden = [111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111,
              333, 333, 333, 333, 333, 333, 333, 333, 333, 333, 333]

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        assert item["label"] == golden[num_iter]
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 22


def test_imagefolder_negative_classindex():
    logger.info("Test Case negative classIndex")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    class_index = {"class3": -333, "class1": 111}
    data1 = ds.ImageFolderDatasetV2(DATA_DIR, class_indexing=class_index, shuffle=False)
    data1 = data1.repeat(repeat_count)

    golden = [111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111,
              -333, -333, -333, -333, -333, -333, -333, -333, -333, -333, -333]

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        assert item["label"] == golden[num_iter]
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 22


def test_imagefolder_extensions():
    logger.info("Test Case extensions")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    ext = [".jpg", ".JPEG"]
    data1 = ds.ImageFolderDatasetV2(DATA_DIR, extensions=ext)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 44


def test_imagefolder_decode():
    logger.info("Test Case decode")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    ext = [".jpg", ".JPEG"]
    data1 = ds.ImageFolderDatasetV2(DATA_DIR, extensions=ext, decode=True)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 44


def test_sequential_sampler():
    logger.info("Test Case SequentialSampler")

    golden = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
              3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

    # define parameters
    repeat_count = 1

    # apply dataset operations
    sampler = ds.SequentialSampler()
    data1 = ds.ImageFolderDatasetV2(DATA_DIR, sampler=sampler)
    data1 = data1.repeat(repeat_count)

    result = []
    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        result.append(item["label"])
        num_iter += 1

    logger.info("Result: {}".format(result))
    assert result == golden


def test_random_sampler():
    logger.info("Test Case RandomSampler")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    sampler = ds.RandomSampler()
    data1 = ds.ImageFolderDatasetV2(DATA_DIR, sampler=sampler)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 44


def test_distributed_sampler():
    logger.info("Test Case DistributedSampler")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    sampler = ds.DistributedSampler(10, 1)
    data1 = ds.ImageFolderDatasetV2(DATA_DIR, sampler=sampler)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 5


def test_pk_sampler():
    logger.info("Test Case PKSampler")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    sampler = ds.PKSampler(3)
    data1 = ds.ImageFolderDatasetV2(DATA_DIR, sampler=sampler)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 12


def test_subset_random_sampler():
    logger.info("Test Case SubsetRandomSampler")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    indices = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 11]
    sampler = ds.SubsetRandomSampler(indices)
    data1 = ds.ImageFolderDatasetV2(DATA_DIR, sampler=sampler)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 12


def test_weighted_random_sampler():
    logger.info("Test Case WeightedRandomSampler")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    weights = [1.0, 0.1, 0.02, 0.3, 0.4, 0.05, 1.2, 0.13, 0.14, 0.015, 0.16, 1.1]
    sampler = ds.WeightedRandomSampler(weights, 11)
    data1 = ds.ImageFolderDatasetV2(DATA_DIR, sampler=sampler)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 11


def test_imagefolder_rename():
    logger.info("Test Case rename")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    data1 = ds.ImageFolderDatasetV2(DATA_DIR, num_samples=10)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 10

    data1 = data1.rename(input_columns=["image"], output_columns="image2")

    num_iter = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image2"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 10


def test_imagefolder_zip():
    logger.info("Test Case zip")
    # define parameters
    repeat_count = 2

    # apply dataset operations
    data1 = ds.ImageFolderDatasetV2(DATA_DIR, num_samples=10)
    data2 = ds.ImageFolderDatasetV2(DATA_DIR, num_samples=10)

    data1 = data1.repeat(repeat_count)
    # rename dataset2 for no conflict
    data2 = data2.rename(input_columns=["image", "label"], output_columns=["image1", "label1"])
    data3 = ds.zip((data1, data2))

    num_iter = 0
    for item in data3.create_dict_iterator():  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 10


if __name__ == '__main__':
    test_imagefolder_basic()
    logger.info('test_imagefolder_basic Ended.\n')

    test_imagefolder_numsamples()
    logger.info('test_imagefolder_numsamples Ended.\n')

    test_sequential_sampler()
    logger.info('test_sequential_sampler Ended.\n')

    test_random_sampler()
    logger.info('test_random_sampler Ended.\n')

    test_distributed_sampler()
    logger.info('test_distributed_sampler Ended.\n')

    test_pk_sampler()
    logger.info('test_pk_sampler Ended.\n')

    test_subset_random_sampler()
    logger.info('test_subset_random_sampler Ended.\n')

    test_weighted_random_sampler()
    logger.info('test_weighted_random_sampler Ended.\n')

    test_imagefolder_numshards()
    logger.info('test_imagefolder_numshards Ended.\n')

    test_imagefolder_shardid()
    logger.info('test_imagefolder_shardid Ended.\n')

    test_imagefolder_noshuffle()
    logger.info('test_imagefolder_noshuffle Ended.\n')

    test_imagefolder_extrashuffle()
    logger.info('test_imagefolder_extrashuffle Ended.\n')

    test_imagefolder_classindex()
    logger.info('test_imagefolder_classindex Ended.\n')

    test_imagefolder_negative_classindex()
    logger.info('test_imagefolder_negative_classindex Ended.\n')

    test_imagefolder_extensions()
    logger.info('test_imagefolder_extensions Ended.\n')

    test_imagefolder_decode()
    logger.info('test_imagefolder_decode Ended.\n')

    test_imagefolder_rename()
    logger.info('test_imagefolder_rename Ended.\n')

    test_imagefolder_zip()
    logger.info('test_imagefolder_zip Ended.\n')
