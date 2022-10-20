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
import pytest
import mindspore.dataset as ds
from mindspore import log as logger

DATA_DIR = "../data/dataset/testIMDBDataset"


def test_imdb_basic():
    """
    Feature: Test IMDB Dataset.
    Description: Read data from all file.
    Expectation: The data is processed successfully.
    """
    logger.info("Test Case basic")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    data1 = ds.IMDBDataset(DATA_DIR, shuffle=False)
    data1 = data1.repeat(repeat_count)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 8

    content = ["train_pos_0.txt", "train_pos_1.txt", "train_neg_0.txt", "train_neg_1.txt",
               "test_pos_0.txt", "test_pos_1.txt", "test_neg_0.txt", "test_neg_1.txt"]
    label = [1, 1, 0, 0, 1, 1, 0, 0]

    num_iter = 0
    for index, item in enumerate(data1.create_dict_iterator(num_epochs=1, output_numpy=True)):
        # each data is a dictionary
        # in this example, each dictionary has keys "text" and "label"
        strs = item["text"]
        logger.info("text is {}".format(strs))
        logger.info("label is {}".format(item["label"]))
        assert strs == content[index]
        assert label[index] == int(item["label"])
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 8


def test_imdb_test():
    """
    Feature: Test IMDB Dataset.
    Description: Read data from test file.
    Expectation: The data is processed successfully.
    """
    logger.info("Test Case test")
    # define parameters
    repeat_count = 1
    usage = "test"
    # apply dataset operations
    data1 = ds.IMDBDataset(DATA_DIR, usage=usage, shuffle=False)
    data1 = data1.repeat(repeat_count)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 4

    content = ["test_pos_0.txt", "test_pos_1.txt", "test_neg_0.txt", "test_neg_1.txt"]
    label = [1, 1, 0, 0]

    num_iter = 0
    for index, item in enumerate(data1.create_dict_iterator(num_epochs=1, output_numpy=True)):
        # each data is a dictionary
        # in this example, each dictionary has keys "text" and "label"
        strs = item["text"]
        logger.info("text is {}".format(strs))
        logger.info("label is {}".format(item["label"]))
        assert strs == content[index]
        assert label[index] == int(item["label"])
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 4


def test_imdb_train():
    """
    Feature: Test IMDB Dataset.
    Description: Read data from train file.
    Expectation: The data is processed successfully.
    """
    logger.info("Test Case train")
    # define parameters
    repeat_count = 1
    usage = "train"
    # apply dataset operations
    data1 = ds.IMDBDataset(DATA_DIR, usage=usage, shuffle=False)
    data1 = data1.repeat(repeat_count)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 4

    content = ["train_pos_0.txt", "train_pos_1.txt", "train_neg_0.txt", "train_neg_1.txt"]
    label = [1, 1, 0, 0]

    num_iter = 0
    for index, item in enumerate(data1.create_dict_iterator(num_epochs=1, output_numpy=True)):
        # each data is a dictionary
        # in this example, each dictionary has keys "text" and "label"
        strs = item["text"]
        logger.info("text is {}".format(strs))
        logger.info("label is {}".format(item["label"]))
        assert strs == content[index]
        assert label[index] == int(item["label"])
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 4


def test_imdb_num_samples():
    """
    Feature: Test IMDB Dataset.
    Description: Read data from all file with num_samples=10 and num_parallel_workers=2.
    Expectation: The data is processed successfully.
    """
    logger.info("Test Case numSamples")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    data1 = ds.IMDBDataset(DATA_DIR, num_samples=6, num_parallel_workers=2)
    data1 = data1.repeat(repeat_count)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 6

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "text" and "label"
        logger.info("text is {}".format(item["text"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 6

    random_sampler = ds.RandomSampler(num_samples=3, replacement=True)
    data1 = ds.IMDBDataset(DATA_DIR, num_parallel_workers=2, sampler=random_sampler)

    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1

    assert num_iter == 3

    random_sampler = ds.RandomSampler(num_samples=3, replacement=False)
    data1 = ds.IMDBDataset(DATA_DIR, num_parallel_workers=2, sampler=random_sampler)

    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1

    assert num_iter == 3


def test_imdb_num_shards():
    """
    Feature: Test IMDB Dataset.
    Description: Read data from all file with num_shards=2 and shard_id=1.
    Expectation: The data is processed successfully.
    """
    logger.info("Test Case numShards")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    data1 = ds.IMDBDataset(DATA_DIR, num_shards=2, shard_id=1)
    data1 = data1.repeat(repeat_count)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 4

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "text" and "label"
        logger.info("text is {}".format(item["text"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 4


def test_imdb_shard_id():
    """
    Feature: Test IMDB Dataset.
    Description: Read data from all file with num_shards=4 and shard_id=1.
    Expectation: The data is processed successfully.
    """
    logger.info("Test Case withShardID")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    data1 = ds.IMDBDataset(DATA_DIR, num_shards=2, shard_id=0)
    data1 = data1.repeat(repeat_count)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 4

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "text" and "label"
        logger.info("text is {}".format(item["text"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 4


def test_imdb_no_shuffle():
    """
    Feature: Test IMDB Dataset.
    Description: Read data from all file with shuffle=False.
    Expectation: The data is processed successfully.
    """
    logger.info("Test Case noShuffle")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    data1 = ds.IMDBDataset(DATA_DIR, shuffle=False)
    data1 = data1.repeat(repeat_count)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 8

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "text" and "label"
        logger.info("text is {}".format(item["text"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 8


def test_imdb_true_shuffle():
    """
    Feature: Test IMDB Dataset.
    Description: Read data from all file with shuffle=True.
    Expectation: The data is processed successfully.
    """
    logger.info("Test Case extraShuffle")
    # define parameters
    repeat_count = 2

    # apply dataset operations
    data1 = ds.IMDBDataset(DATA_DIR, shuffle=True)
    data1 = data1.repeat(repeat_count)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 16

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "text" and "label"
        logger.info("text is {}".format(item["text"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 16


def test_random_sampler():
    """
    Feature: Test IMDB Dataset.
    Description: Read data from all file with sampler=ds.RandomSampler().
    Expectation: The data is processed successfully.
    """
    logger.info("Test Case RandomSampler")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    sampler = ds.RandomSampler()
    data1 = ds.IMDBDataset(DATA_DIR, sampler=sampler)
    data1 = data1.repeat(repeat_count)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 8

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "text" and "label"
        logger.info("text is {}".format(item["text"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 8


def test_distributed_sampler():
    """
    Feature: Test IMDB Dataset.
    Description: Read data from all file with sampler=ds.DistributedSampler().
    Expectation: The data is processed successfully.
    """
    logger.info("Test Case DistributedSampler")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    sampler = ds.DistributedSampler(4, 1)
    data1 = ds.IMDBDataset(DATA_DIR, sampler=sampler)
    data1 = data1.repeat(repeat_count)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 2

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "text" and "label"
        logger.info("text is {}".format(item["text"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 2


def test_pk_sampler():
    """
    Feature: Test IMDB Dataset.
    Description: Read data from all file with sampler=ds.PKSampler().
    Expectation: The data is processed successfully.
    """
    logger.info("Test Case PKSampler")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    sampler = ds.PKSampler(3)
    data1 = ds.IMDBDataset(DATA_DIR, sampler=sampler)
    data1 = data1.repeat(repeat_count)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 6

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "text" and "label"
        logger.info("text is {}".format(item["text"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 6


def test_subset_random_sampler():
    """
    Feature: Test IMDB Dataset.
    Description: Read data from all file with sampler=ds.SubsetRandomSampler().
    Expectation: The data is processed successfully.
    """
    logger.info("Test Case SubsetRandomSampler")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    indices = [0, 3, 1, 2, 5, 4]
    sampler = ds.SubsetRandomSampler(indices)
    data1 = ds.IMDBDataset(DATA_DIR, sampler=sampler)
    data1 = data1.repeat(repeat_count)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 6

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "text" and "label"
        logger.info("text is {}".format(item["text"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 6


def test_weighted_random_sampler():
    """
    Feature: Test IMDB Dataset.
    Description: Read data from all file with sampler=ds.WeightedRandomSampler().
    Expectation: The data is processed successfully.
    """
    logger.info("Test Case WeightedRandomSampler")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    weights = [1.0, 0.1, 0.02, 0.3, 0.4, 0.05]
    sampler = ds.WeightedRandomSampler(weights, 6)
    data1 = ds.IMDBDataset(DATA_DIR, sampler=sampler)
    data1 = data1.repeat(repeat_count)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 6

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "text" and "label"
        logger.info("text is {}".format(item["text"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 6


def test_weighted_random_sampler_exception():
    """
    Feature: Test IMDB Dataset.
    Description: Read data from all file with random sampler exception.
    Expectation: The data is processed successfully.
    """
    logger.info("Test error cases for WeightedRandomSampler")
    error_msg_1 = "type of weights element must be number"
    with pytest.raises(TypeError, match=error_msg_1):
        weights = ""
        ds.WeightedRandomSampler(weights)

    error_msg_2 = "type of weights element must be number"
    with pytest.raises(TypeError, match=error_msg_2):
        weights = (0.9, 0.8, 1.1)
        ds.WeightedRandomSampler(weights)

    error_msg_3 = "WeightedRandomSampler: weights vector must not be empty"
    with pytest.raises(RuntimeError, match=error_msg_3):
        weights = []
        sampler = ds.WeightedRandomSampler(weights)
        sampler.parse()

    error_msg_4 = "WeightedRandomSampler: weights vector must not contain negative numbers, got: "
    with pytest.raises(RuntimeError, match=error_msg_4):
        weights = [1.0, 0.1, 0.02, 0.3, -0.4]
        sampler = ds.WeightedRandomSampler(weights)
        sampler.parse()

    error_msg_5 = "WeightedRandomSampler: elements of weights vector must not be all zero"
    with pytest.raises(RuntimeError, match=error_msg_5):
        weights = [0, 0, 0, 0, 0]
        sampler = ds.WeightedRandomSampler(weights)
        sampler.parse()


def test_chained_sampler_with_random_sequential_repeat():
    """
    Feature: Test IMDB Dataset.
    Description: Read data from all file with Random and Sequential, with repeat.
    Expectation: The data is processed successfully.
    """
    logger.info("Test Case Chained Sampler - Random and Sequential, with repeat")

    # Create chained sampler, random and sequential
    sampler = ds.RandomSampler()
    child_sampler = ds.SequentialSampler()
    sampler.add_child(child_sampler)
    # Create IMDBDataset with sampler
    data1 = ds.IMDBDataset(DATA_DIR, sampler=sampler)

    data1 = data1.repeat(count=3)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 24

    # Verify number of iterations
    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "text" and "label"
        logger.info("text is {}".format(item["text"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 24


def test_chained_sampler_with_distribute_random_batch_then_repeat():
    """
    Feature: Test IMDB Dataset.
    Description: Read data from all file with Distributed and Random, with batch then repeat.
    Expectation: The data is processed successfully.
    """
    logger.info("Test Case Chained Sampler - Distributed and Random, with batch then repeat")

    # Create chained sampler, distributed and random
    sampler = ds.DistributedSampler(num_shards=4, shard_id=3)
    child_sampler = ds.RandomSampler()
    sampler.add_child(child_sampler)
    # Create IMDBDataset with sampler
    data1 = ds.IMDBDataset(DATA_DIR, sampler=sampler)

    data1 = data1.batch(batch_size=5, drop_remainder=True)
    data1 = data1.repeat(count=3)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 0

    # Verify number of iterations
    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "text" and "label"
        logger.info("text is {}".format(item["text"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    # Note: Each of the 4 shards has 44/4=11 samples
    # Note: Number of iterations is (11/5 = 2) * 3 = 6
    assert num_iter == 0


def test_chained_sampler_with_weighted_random_pk_sampler():
    """
    Feature: Test IMDB Dataset.
    Description: Read data from all file with WeightedRandom and PKSampler.
    Expectation: The data is processed successfully.
    """
    logger.info("Test Case Chained Sampler - WeightedRandom and PKSampler")

    # Create chained sampler, WeightedRandom and PKSampler
    weights = [1.0, 0.1, 0.02, 0.3, 0.4, 0.05]
    sampler = ds.WeightedRandomSampler(weights=weights, num_samples=6)
    child_sampler = ds.PKSampler(num_val=3)  # Number of elements per class is 3 (and there are 4 classes)
    sampler.add_child(child_sampler)
    # Create IMDBDataset with sampler
    data1 = ds.IMDBDataset(DATA_DIR, sampler=sampler)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 6

    # Verify number of iterations
    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "text" and "label"
        logger.info("text is {}".format(item["text"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    # Note: WeightedRandomSampler produces 12 samples
    # Note: Child PKSampler produces 12 samples
    assert num_iter == 6


def test_imdb_rename():
    """
    Feature: Test IMDB Dataset.
    Description: Read data from all file with rename.
    Expectation: The data is processed successfully.
    """
    logger.info("Test Case rename")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    data1 = ds.IMDBDataset(DATA_DIR, num_samples=8)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "text" and "label"
        logger.info("text is {}".format(item["text"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 8

    data1 = data1.rename(input_columns=["text"], output_columns="text2")

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "text" and "label"
        logger.info("text is {}".format(item["text2"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 8


def test_imdb_zip():
    """
    Feature: Test IMDB Dataset.
    Description: Read data from all file with zip.
    Expectation: The data is processed successfully.
    """
    logger.info("Test Case zip")
    # define parameters
    repeat_count = 2

    # apply dataset operations
    data1 = ds.IMDBDataset(DATA_DIR, num_samples=4)
    data2 = ds.IMDBDataset(DATA_DIR, num_samples=4)

    data1 = data1.repeat(repeat_count)
    # rename dataset2 for no conflict
    data2 = data2.rename(input_columns=["text", "label"], output_columns=["text1", "label1"])
    data3 = ds.zip((data1, data2))

    num_iter = 0
    for item in data3.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "text" and "label"
        logger.info("text is {}".format(item["text"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 4


def test_imdb_exception():
    """
    Feature: Test IMDB Dataset.
    Description: Read data from all file with exception.
    Expectation: The data is processed successfully.
    """
    logger.info("Test imdb exception")

    def exception_func(item):
        raise Exception("Error occur!")

    def exception_func2(text, label):
        raise Exception("Error occur!")

    try:
        data = ds.IMDBDataset(DATA_DIR)
        data = data.map(operations=exception_func, input_columns=["text"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.IMDBDataset(DATA_DIR)
        data = data.map(operations=exception_func2, input_columns=["text", "label"],
                        output_columns=["text", "label", "label1"],
                        num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    data_dir_invalid = "../data/dataset/IMDBDATASET"
    try:
        data = ds.IMDBDataset(data_dir_invalid)
        for _ in data.__iter__():
            pass
        assert False
    except ValueError as e:
        assert "does not exist or is not a directory or permission denied" in str(e)


if __name__ == '__main__':
    test_imdb_basic()
    test_imdb_test()
    test_imdb_train()
    test_imdb_num_samples()
    test_random_sampler()
    test_distributed_sampler()
    test_pk_sampler()
    test_subset_random_sampler()
    test_weighted_random_sampler()
    test_weighted_random_sampler_exception()
    test_chained_sampler_with_random_sequential_repeat()
    test_chained_sampler_with_distribute_random_batch_then_repeat()
    test_chained_sampler_with_weighted_random_pk_sampler()
    test_imdb_num_shards()
    test_imdb_shard_id()
    test_imdb_no_shuffle()
    test_imdb_true_shuffle()
    test_imdb_rename()
    test_imdb_zip()
    test_imdb_exception()
