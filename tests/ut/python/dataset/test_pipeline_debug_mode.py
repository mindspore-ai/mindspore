# Copyright 2022 Huawei Technologies Co., Ltd
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
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Inter
from mindspore import log as logger

# Need to run all these tests in separate processes since
# the global configuration setting of debug_mode may impact other tests running in parallel.
pytestmark = pytest.mark.forked

DATA_DIR_10 = "../data/dataset/testCifar10Data"
DEBUG_MODE = False
SEED_VAL = 0  # seed will be set internally in debug mode, save original seed value to restore.


def setup_function():
    global DEBUG_MODE
    global SEED_VAL
    DEBUG_MODE = ds.config.get_debug_mode()
    SEED_VAL = ds.config.get_seed()
    ds.config.set_debug_mode(True)


def teardown_function():
    ds.config.set_debug_mode(DEBUG_MODE)
    ds.config.set_seed(SEED_VAL)


def test_pipeline_debug_mode_tuple():
    """
    Feature: Pipeline debug mode.
    Description: Test creating tuple iterator with debug mode enabled.
    Expectation: Successful.
    """
    logger.info("test_pipeline_debug_mode_tuple")
    data = ds.CelebADataset("../data/dataset/testCelebAData/", decode=True, num_shards=1, shard_id=0)
    crop_size = (80, 80)
    resize_size = (24, 24)
    # define map operations
    center_crop = vision.CenterCrop(crop_size)
    resize_op = vision.Resize(resize_size, Inter.LINEAR)  # Bilinear mode
    data = data.map(operations=[center_crop, resize_op], input_columns=["image"])
    data = data.batch(2)
    num_row = 0
    for item in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 2
        assert item[0].shape == (2, 24, 24, 3)
        assert item[1].shape == (2, 40)
        num_row += 1
    assert num_row == 2


def test_pipeline_debug_mode_dict():
    """
    Feature: Pipeline debug mode.
    Description: Test creating dict iterator with debug mode enabled.
    Expectation: Successful.
    """
    logger.info("test_pipeline_debug_mode_dict")
    data = ds.CelebADataset("../data/dataset/testCelebAData/", decode=True, num_shards=1, shard_id=0)
    crop_size = (80, 80)
    resize_size = (24, 24)
    # define map operations
    center_crop = vision.CenterCrop(crop_size)
    resize_op = vision.Resize(resize_size, Inter.LINEAR)  # Bilinear mode
    data = data.map(operations=[center_crop, resize_op], input_columns=["image"])
    data = data.batch(2)
    num_row = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 2
        assert item["image"].shape == (2, 24, 24, 3)
        assert item["attr"].shape == (2, 40)
        num_row += 1
    assert num_row == 2


def test_pipeline_debug_mode_minddata():
    """
    Feature: Pipeline debug mode.
    Description: Test iterator with MindDataset in debug mode.
    Expectation:Successful.
    """
    logger.info("test_pipeline_debug_mode_minddata")
    data = ds.MindDataset("../data/mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0")
    data_lst = []
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 3
        data_lst.append(item["data"].copy())
    assert len(data_lst) == 20


def test_pipeline_debug_mode_not_support():
    """
    Feature: Pipeline debug mode.
    Description: Test creating tuple iterator with op not supported in pull mode.
    Expectation: raise exception for debug mode.
    """
    logger.info("test_pipeline_debug_mode_not_support")
    data = ds.NumpySlicesDataset(data=[[0, 1, 2]], column_names=["data"])
    with pytest.raises(RuntimeError) as error_info:
        data.create_tuple_iterator(num_epochs=1, output_numpy=True)
    assert "dataset pipeline" in str(error_info.value)


def test_pipeline_debug_mode_map_pyfunc():
    """
    Feature: Pipeline debug mode.
    Description: Test creating dict iterator with map(PyFunc).
    Expectation: Successful.
    """
    logger.info("test_pipeline_debug_mode_map_pyfunc")
    data = ds.CelebADataset("../data/dataset/testCelebAData/", decode=True, num_shards=1, shard_id=0)
    data = data.map(operations=[(lambda x: x - 1), (lambda x: x * 2)], input_columns=["image"])
    num_rows = 0
    for item in data.create_dict_iterator(num_epochs=1):
        assert len(item) == 2
        assert item["image"].shape == (2268, 4032, 3)
        num_rows += 1
    assert num_rows == 4


def test_pipeline_debug_mode_batch_pyfunc():
    """
    Feature: Pipeline debug mode.
    Description: Test creating dict iterator with Batch(PyFunc).
    Expectation: Successful.
    """
    logger.info("test_pipeline_debug_mode_batch_pyfunc")

    def add_one(batch_info):
        return batch_info.get_batch_num() + 1

    data = ds.MnistDataset("../data/dataset/testMnistData", num_samples=20)
    data = data.batch(batch_size=add_one, drop_remainder=True)
    num_rows = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_rows += 1
        assert item["label"].shape == (num_rows,)
    assert num_rows == 5


def test_pipeline_debug_mode_concat():
    """
    Feature: Pipeline debug mode.
    Description: Test creating tuple iterator with concat.
    Expectation: Successful.
    """
    logger.info("test_pipeline_debug_mode_concat")
    data_dir = "../data/dataset/testCelebAData/"
    data1 = ds.CelebADataset(data_dir, decode=True, num_shards=1, shard_id=0)
    data2 = ds.CelebADataset(data_dir, decode=True, num_shards=1, shard_id=0)
    data3 = ds.CelebADataset(data_dir, decode=True, num_shards=1, shard_id=0)
    data4 = data1.concat(data2)
    data5 = data3 + data4
    num_rows = 0
    for item1 in data5.create_tuple_iterator(num_epochs=1):
        assert len(item1) == 2
        assert item1[0].shape == (2268, 4032, 3)
        num_rows += 1
    assert num_rows == 12


def test_pipeline_debug_mode_map_random():
    """
    Feature: Pipeline debug mode.
    Description: Test creating dict iterator with map with random augmentation operations.
    Expectation: Successful.
    """
    logger.info("test_pipeline_debug_mode_map_random")
    # the explicit intent of this test to not set the seed and allow debug mode support to set it
    #   (if the default seed is used)
    data = ds.CelebADataset("../data/dataset/testCelebAData/", decode=True, num_shards=1, shard_id=0)
    transforms_list = [vision.CenterCrop(64), vision.RandomRotation(30)]
    random_apply = transforms.RandomApply(transforms_list, prob=0.6)

    data = data.map(operations=[random_apply], input_columns=["image"])
    expected_shape = [(2268, 4032, 3), (2268, 4032, 3), (64, 64, 3), (2268, 4032, 3)]
    index = 0
    for item in data.create_dict_iterator(num_epochs=1):
        assert len(item) == 2
        assert item["image"].shape == expected_shape[index]
        index += 1
    assert index == 4


def test_pipeline_debug_mode_shuffle():
    """
    Feature: Pipeline debug mode.
    Description: Test creating dict iterator with Shuffle.
    Expectation: Shuffle is disabled, but has the same number of rows as not in debug mode.
    """
    logger.info("test_pipeline_debug_mode_shuffle")

    buffer_size = 5
    data = ds.MnistDataset("../data/dataset/testMnistData", num_samples=20)
    data = data.shuffle(buffer_size=buffer_size)
    num_rows = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_rows += 1
    assert num_rows == 20


def test_pipeline_debug_mode_imdb_shuffle():
    """
    Feature: Pipeline debug mode.
    Description: Justify shuffle is disabled with IMDBDataset
    Expectation: The data is processed successfully in the same order.
    """
    logger.info("test_pipeline_debug_mode_imdb_shuffle")
    buffer_size = 5

    # apply dataset operations
    data1 = ds.IMDBDataset("../data/dataset/testIMDBDataset", shuffle=True)
    data1 = data1.shuffle(buffer_size=buffer_size)

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


if __name__ == '__main__':
    setup_function()
    test_pipeline_debug_mode_tuple()
    test_pipeline_debug_mode_dict()
    test_pipeline_debug_mode_minddata()
    test_pipeline_debug_mode_not_support()
    test_pipeline_debug_mode_map_pyfunc()
    test_pipeline_debug_mode_batch_pyfunc()
    test_pipeline_debug_mode_concat()
    test_pipeline_debug_mode_shuffle()
    test_pipeline_debug_mode_map_random()
    test_pipeline_debug_mode_imdb_shuffle()
    teardown_function()
