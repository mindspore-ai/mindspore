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

import re
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger

DATA_DIR = "../data/dataset/testKITTI"
IMAGE_SHAPE = [2268, 642, 2268]


def test_func_kitti_dataset_basic():
    """
    Feature: KITTI
    Description: Test basic function of KITTI with default parament
    Expectation: The dataset is as expected
    """
    repeat_count = 2

    # apply dataset operations.
    data = ds.KITTIDataset(DATA_DIR, shuffle=False)
    data = data.repeat(repeat_count)

    num_iter = 0
    count = [0, 0, 0, 0, 0, 0, 0, 0]
    SHAPE = [159109, 176455, 54214, 159109, 176455, 54214]
    ANNOTATIONSHAPE = [6, 3, 7, 6, 3, 7]
    # each data is a dictionary.
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image", "label", "truncated", "occluded", "alpha", "bbox",
        #     "dimensions", "location", "rotation_y".
        assert item["image"].shape[0] == SHAPE[num_iter]
        for label in item["label"]:
            count[label[0]] += 1
        assert item["truncated"].shape[0] == ANNOTATIONSHAPE[num_iter]
        assert item["occluded"].shape[0] == ANNOTATIONSHAPE[num_iter]
        assert item["alpha"].shape[0] == ANNOTATIONSHAPE[num_iter]
        assert item["bbox"].shape[0] == ANNOTATIONSHAPE[num_iter]
        assert item["dimensions"].shape[0] == ANNOTATIONSHAPE[num_iter]
        assert item["location"].shape[0] == ANNOTATIONSHAPE[num_iter]
        assert item["rotation_y"].shape[0] == ANNOTATIONSHAPE[num_iter]
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 6
    assert count == [8, 20, 2, 2, 0, 0, 0, 0]


def test_kitti_usage_train():
    """
    Feature: KITTI
    Description: Test basic usage "train" of KITTI
    Expectation: The dataset is as expected
    """
    data1 = ds.KITTIDataset(DATA_DIR, usage="train")
    num = 0
    count = [0, 0, 0, 0, 0, 0, 0, 0]
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        for label in item["label"]:
            count[label[0]] += 1
        num += 1
    assert num == 3
    assert count == [4, 10, 1, 1, 0, 0, 0, 0]


def test_kitti_usage_test():
    """
    Feature: KITTI
    Description: Test basic usage "test" of KITTI
    Expectation: The dataset is as expected
    """
    data1 = ds.KITTIDataset(
        DATA_DIR, usage="test", shuffle=False, decode=True, num_samples=3)
    num = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert item["image"].shape[0] == IMAGE_SHAPE[num]
        num += 1
    assert num == 3


def test_kitti_case():
    """
    Feature: KITTI
    Description: Test basic usage of KITTI
    Expectation: The dataset is as expected
    """
    data1 = ds.KITTIDataset(DATA_DIR,
                            usage="train", decode=True, num_samples=3)
    resize_op = vision.Resize((224, 224))
    data1 = data1.map(operations=resize_op, input_columns=["image"])
    repeat_num = 4
    data1 = data1.repeat(repeat_num)
    batch_size = 2
    data1 = data1.padded_batch(batch_size, drop_remainder=True, pad_info={})
    num = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num += 1
    assert num == 6


def test_func_kitti_dataset_numsamples_num_parallel_workers():
    """
    Feature: KITTI
    Description: Test numsamples and num_parallel_workers of KITTI
    Expectation: The dataset is as expected
    """
    # define parameters.
    repeat_count = 2

    # apply dataset operations.
    data1 = ds.KITTIDataset(DATA_DIR, num_samples=2, num_parallel_workers=2)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 4

    random_sampler = ds.RandomSampler(num_samples=3, replacement=True)
    data1 = ds.KITTIDataset(DATA_DIR, num_parallel_workers=2,
                            sampler=random_sampler)

    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1

    assert num_iter == 3

    random_sampler = ds.RandomSampler(num_samples=3, replacement=False)
    data1 = ds.KITTIDataset(DATA_DIR, num_parallel_workers=2,
                            sampler=random_sampler)

    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1

    assert num_iter == 3


def test_func_kitti_dataset_extrashuffle():
    """
    Feature: KITTI
    Description: Test extrashuffle of KITTI
    Expectation: The dataset is as expected
    """
    # define parameters.
    repeat_count = 2

    # apply dataset operations.
    data1 = ds.KITTIDataset(DATA_DIR, shuffle=True)
    data1 = data1.shuffle(buffer_size=3)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 6


def test_func_kitti_dataset_no_para():
    """
    Feature: KITTI
    Description: Test no para of KITTI
    Expectation: Throw exception correctly
    """
    with pytest.raises(TypeError, match="missing a required argument: 'dataset_dir'"):
        dataset = ds.KITTIDataset()
        num_iter = 0
        for data in dataset.create_dict_iterator(output_numpy=True):
            assert "image" in str(data.keys())
            num_iter += 1


def test_func_kitti_dataset_distributed_sampler():
    """
    Feature: KITTI
    Description: Test DistributedSampler of KITTI
    Expectation: Throw exception correctly
    """
    # define parameters.
    repeat_count = 2

    # apply dataset operations.
    sampler = ds.DistributedSampler(3, 1)
    data1 = ds.KITTIDataset(DATA_DIR, sampler=sampler)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 2


def test_func_kitti_dataset_decode():
    """
    Feature: KITTI
    Description: Test decode of KITTI
    Expectation: Throw exception correctly
    """
    # define parameters.
    repeat_count = 2

    # apply dataset operations.
    data1 = ds.KITTIDataset(DATA_DIR, decode=True)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 6


def test_kitti_numshards():
    """
    Feature: KITTI
    Description: Test numShards of KITTI
    Expectation: Throw exception correctly
    """
    # define parameters.
    repeat_count = 2

    # apply dataset operations.
    data1 = ds.KITTIDataset(DATA_DIR, num_shards=3, shard_id=2)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 2


def test_func_kitti_dataset_more_para():
    """
    Feature: KITTI
    Description: Test more para of KITTI
    Expectation: Throw exception correctly
    """
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'more_para'"):
        dataset = ds.KITTIDataset(DATA_DIR, usage="train", num_samples=6, num_parallel_workers=None,
                                  shuffle=True, sampler=None, decode=True, num_shards=3,
                                  shard_id=2, cache=None, more_para=None)
        num_iter = 0
        for data in dataset.create_dict_iterator(output_numpy=True):
            num_iter += 1
            assert "image" in str(data.keys())


def test_kitti_exception():
    """
    Feature: KITTI
    Description: Test error cases of KITTI
    Expectation: Throw exception correctly
    """
    logger.info("Test error cases for KITTIDataset")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.KITTIDataset(DATA_DIR, shuffle=False, decode=True, sampler=ds.SequentialSampler(1))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.KITTIDataset(DATA_DIR, sampler=ds.SequentialSampler(1), decode=True, num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.KITTIDataset(DATA_DIR, decode=True, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.KITTIDataset(DATA_DIR, decode=True, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.KITTIDataset(DATA_DIR, decode=True, num_shards=5, shard_id=-1)

    with pytest.raises(ValueError, match=error_msg_5):
        ds.KITTIDataset(DATA_DIR, decode=True, num_shards=5, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.KITTIDataset(DATA_DIR, decode=True, shuffle=False, num_parallel_workers=0)

    with pytest.raises(ValueError, match=error_msg_6):
        ds.KITTIDataset(DATA_DIR, decode=True, shuffle=False, num_parallel_workers=256)

    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.KITTIDataset(DATA_DIR, decode=True, num_shards=2, shard_id="0")

    error_msg_8 = "does not exist or is not a directory or permission denied!"
    with pytest.raises(ValueError, match=error_msg_8):
        all_data = ds.KITTIDataset("../data/dataset/testKITTI2", decode=True)
        for _ in all_data.create_dict_iterator(num_epochs=1):
            pass

    error_msg_9 = "Input usage is not within the valid set of ['train', 'test']."
    with pytest.raises(ValueError, match=re.escape(error_msg_9)):
        all_data = ds.KITTIDataset(DATA_DIR, usage="all")
        for _ in all_data.create_dict_iterator(num_epochs=1):
            pass

    error_msg_10 = "Argument decode with value 123 is not of type [<class 'bool'>], but got <class 'int'>."
    with pytest.raises(TypeError, match=re.escape(error_msg_10)):
        all_data = ds.KITTIDataset(DATA_DIR, decode=123)
        for _ in all_data.create_dict_iterator(num_epochs=1):
            pass


if __name__ == '__main__':
    test_func_kitti_dataset_basic()
    test_kitti_usage_train()
    test_kitti_usage_test()
    test_kitti_case()
    test_func_kitti_dataset_numsamples_num_parallel_workers()
    test_func_kitti_dataset_extrashuffle()
    test_func_kitti_dataset_no_para()
    test_func_kitti_dataset_distributed_sampler()
    test_func_kitti_dataset_decode()
    test_kitti_numshards()
    test_func_kitti_dataset_more_para()
    test_kitti_exception()
