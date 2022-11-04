# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
Test Omniglot dataset operations
"""
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import config_get_set_seed

DATA_DIR = "../data/dataset/testOmniglot"


def test_omniglot_basic():
    """
    Feature: load_omniglot.
    Description: Load OmniglotDataset.
    Expectation: Get data of OmniglotDataset.
    """
    logger.info("Test Case basic")
    # define parameters.
    repeat_count = 1

    # apply dataset operations.
    data1 = ds.OmniglotDataset(DATA_DIR)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    count = [0, 0, 0, 0]
    BASIC_EXPECTED_SHAPE = {"82386": 1, "61235": 1, "159109": 2}
    ACTUAL_SHAPE = {"82386": 0, "61235": 0, "159109": 0}
    # each data is a dictionary.
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        ACTUAL_SHAPE[str(item["image"].shape[0])] += 1
        count[item["label"]] += 1
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 4
    assert count == [2, 2, 0, 0]
    assert ACTUAL_SHAPE == BASIC_EXPECTED_SHAPE


def test_omniglot_num_samples():
    """
    Feature: load_omniglot.
    Description: Load OmniglotDataset.
    Expectation: Get data of OmniglotDataset.
    """
    logger.info("Test Case numSamples")
    # define parameters.
    repeat_count = 1

    # apply dataset operations.
    data1 = ds.OmniglotDataset(DATA_DIR, num_samples=8, num_parallel_workers=2)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 4

    random_sampler = ds.RandomSampler(num_samples=3, replacement=True)
    data1 = ds.OmniglotDataset(DATA_DIR,
                               num_parallel_workers=2,
                               sampler=random_sampler)

    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1

    assert num_iter == 3

    random_sampler = ds.RandomSampler(num_samples=3, replacement=False)
    data1 = ds.OmniglotDataset(DATA_DIR,
                               num_parallel_workers=2,
                               sampler=random_sampler)

    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1

    assert num_iter == 3


def test_omniglot_num_shards():
    """
    Feature: load_omniglot.
    Description: Load OmniglotDataset.
    Expectation: Get data of OmniglotDataset.
    """
    logger.info("Test Case numShards")
    # define parameters.
    repeat_count = 1

    original_seed = config_get_set_seed(0)

    # apply dataset operations.
    data1 = ds.OmniglotDataset(DATA_DIR, num_shards=4, shard_id=2)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        assert item["image"].shape[0] == 82386
        assert item["label"] == 1
        num_iter += 1
    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 1

    ds.config.set_seed(original_seed)


def test_omniglot_shard_id():
    """
    Feature: load_omniglot.
    Description: Load OmniglotDataset.
    Expectation: Get data of OmniglotDataset.
    """
    logger.info("Test Case withShardID")
    # define parameters.
    repeat_count = 1

    original_seed = config_get_set_seed(0)

    # apply dataset operations.
    data1 = ds.OmniglotDataset(DATA_DIR, num_shards=4, shard_id=1)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        assert item["image"].shape[0] == 159109
        assert item["label"] == 0
        num_iter += 1
    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 1

    ds.config.set_seed(original_seed)


def test_omniglot_no_shuffle():
    """
    Feature: load_omniglot.
    Description: Load OmniglotDataset.
    Expectation: Get data of OmniglotDataset.
    """
    logger.info("Test Case noShuffle")
    # define parameters.
    repeat_count = 1

    # apply dataset operations.
    data1 = ds.OmniglotDataset(DATA_DIR, shuffle=False)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    count = [0, 0, 0, 0]
    SHAPE = [159109, 159109, 82386, 61235]
    # each data is a dictionary.
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        assert item["image"].shape[0] == SHAPE[num_iter]
        count[item["label"]] += 1
        num_iter += 1

    assert num_iter == 4
    assert count == [2, 2, 0, 0]


def test_omniglot_extra_shuffle():
    """
    Feature: load_omniglot.
    Description: Load OmniglotDataset.
    Expectation: Get data of OmniglotDataset.
    """
    logger.info("Test Case extraShuffle")
    # define parameters.
    repeat_count = 2

    # apply dataset operations.
    data1 = ds.OmniglotDataset(DATA_DIR, shuffle=True)
    data1 = data1.shuffle(buffer_size=5)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    count = [0, 0, 0, 0]
    EXPECTED_SHAPE = {"82386": 2, "61235": 2, "159109": 4}
    ACTUAL_SHAPE = {"82386": 0, "61235": 0, "159109": 0}
    # each data is a dictionary.
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        ACTUAL_SHAPE[str(item["image"].shape[0])] += 1
        count[item["label"]] += 1
        num_iter += 1
    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 8
    assert count == [4, 4, 0, 0]
    assert ACTUAL_SHAPE == EXPECTED_SHAPE


def test_omniglot_decode():
    """
    Feature: load_omniglot.
    Description: Load OmniglotDataset.
    Expectation: Get data of OmniglotDataset.
    """
    logger.info("Test Case decode")
    # define parameters.
    repeat_count = 1

    # apply dataset operations.
    data1 = ds.OmniglotDataset(DATA_DIR, decode=True)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 4


def test_sequential_sampler():
    """
    Feature: load_omniglot.
    Description: Load OmniglotDataset.
    Expectation: Get data of OmniglotDataset.
    """
    logger.info("Test Case SequentialSampler")
    # define parameters.
    repeat_count = 1
    # apply dataset operations.
    sampler = ds.SequentialSampler(num_samples=8)
    data1 = ds.OmniglotDataset(DATA_DIR, sampler=sampler)
    data_seq = data1.repeat(repeat_count)

    num_iter = 0
    count = [0, 0, 0, 0]
    SHAPE = [159109, 159109, 82386, 61235]
    # each data is a dictionary.
    for item in data_seq.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        assert item["image"].shape[0] == SHAPE[num_iter]
        count[item["label"]] += 1
        num_iter += 1

    assert num_iter == 4
    assert count == [2, 2, 0, 0]


def test_random_sampler():
    """
    Feature: load_omniglot.
    Description: Load OmniglotDataset.
    Expectation: Get data of OmniglotDataset.
    """
    logger.info("Test Case RandomSampler")
    # define parameters.
    repeat_count = 1

    # apply dataset operations.
    sampler = ds.RandomSampler()
    data1 = ds.OmniglotDataset(DATA_DIR, sampler=sampler)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    count = [0, 0, 0, 0]
    RANDOM_EXPECTED_SHAPE = {"82386": 1, "61235": 1, "159109": 2}
    ACTUAL_SHAPE = {"82386": 0, "61235": 0, "159109": 0}
    # each data is a dictionary.
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        ACTUAL_SHAPE[str(item["image"].shape[0])] += 1
        count[item["label"]] += 1
        num_iter += 1
    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 4
    assert count == [2, 2, 0, 0]
    assert ACTUAL_SHAPE == RANDOM_EXPECTED_SHAPE


def test_distributed_sampler():
    """
    Feature: load_omniglot.
    Description: Load OmniglotDataset.
    Expectation: Get data of OmniglotDataset.
    """
    logger.info("Test Case DistributedSampler")
    # define parameters.
    repeat_count = 1

    original_seed = config_get_set_seed(0)

    # apply dataset operations.
    sampler = ds.DistributedSampler(4, 1)
    data1 = ds.OmniglotDataset(DATA_DIR, sampler=sampler)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        assert item["image"].shape[0] == 159109
        assert item["label"] == 0
        num_iter += 1
    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 1

    ds.config.set_seed(original_seed)


def test_pk_sampler():
    """
    Feature: load_omniglot.
    Description: Load OmniglotDataset.
    Expectation: Get data of OmniglotDataset.
    """
    logger.info("Test Case PKSampler")
    # define parameters.
    repeat_count = 1

    # apply dataset operations.
    sampler = ds.PKSampler(1)
    data1 = ds.OmniglotDataset(DATA_DIR, sampler=sampler)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 2


def test_chained_sampler():
    """
    Feature: load_omniglot.
    Description: Load OmniglotDataset.
    Expectation: Get data of OmniglotDataset.
    """
    logger.info(
        "Test Case Chained Sampler - Random and Sequential, with repeat")

    # Create chained sampler, random and sequential.
    sampler = ds.RandomSampler()
    child_sampler = ds.SequentialSampler()
    sampler.add_child(child_sampler)
    # Create OmniglotDataset with sampler.
    data1 = ds.OmniglotDataset(DATA_DIR, sampler=sampler)

    data1 = data1.repeat(count=3)

    # Verify dataset size.
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 12

    # Verify number of iterations.
    num_iter = 0
    # each data is a dictionary.
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 12


def test_omniglot_evaluation():
    """
    Feature: load_omniglot.
    Description: Load OmniglotDataset.
    Expectation: Get data of OmniglotDataset.
    """
    logger.info("Test Case usage")
    # apply dataset operations.
    data1 = ds.OmniglotDataset(DATA_DIR, background=False, num_samples=6)
    num_iter = 0
    # each data is a dictionary.
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 4


def test_omniglot_zip():
    """
    Feature: load_omniglot.
    Description: Load OmniglotDataset.
    Expectation: Get data of OmniglotDataset.
    """
    logger.info("Test Case zip")
    # define parameters.
    repeat_count = 2

    # apply dataset operations.
    data1 = ds.OmniglotDataset(DATA_DIR, num_samples=8)
    data2 = ds.OmniglotDataset(DATA_DIR, num_samples=8)

    data1 = data1.repeat(repeat_count)
    # rename dataset2 for no conflict.
    data2 = data2.rename(input_columns=["image", "label"],
                         output_columns=["image1", "label1"])
    data3 = ds.zip((data1, data2))

    num_iter = 0
    # each data is a dictionary.
    for _ in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 4


def test_omniglot_exception():
    """
    Feature: test_omniglot_exception.
    Description: Test error cases for OmniglotDataset.
    Expectation: Raise exception.
    """
    logger.info("Test omniglot exception")

    def exception_func(item):
        raise Exception("Error occur!")

    def exception_func2(image, label):
        raise Exception("Error occur!")

    try:
        data = ds.OmniglotDataset(DATA_DIR)
        data = data.map(operations=exception_func,
                        input_columns=["image"],
                        num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(
            e)

    try:
        data = ds.OmniglotDataset(DATA_DIR)
        data = data.map(operations=exception_func2,
                        input_columns=["image", "label"],
                        output_columns=["image", "label", "label1"],
                        num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.OmniglotDataset(DATA_DIR)
        data = data.map(operations=vision.Decode(), input_columns=["image"], num_parallel_workers=1)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)


if __name__ == '__main__':
    test_omniglot_basic()
    test_omniglot_num_samples()
    test_sequential_sampler()
    test_random_sampler()
    test_distributed_sampler()
    test_chained_sampler()
    test_pk_sampler()
    test_omniglot_num_shards()
    test_omniglot_shard_id()
    test_omniglot_no_shuffle()
    test_omniglot_extra_shuffle()
    test_omniglot_decode()
    test_omniglot_evaluation()
    test_omniglot_zip()
    test_omniglot_exception()
