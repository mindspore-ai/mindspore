# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
Test dataset operations in debug mode
"""

import numpy as np
import pytest
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Inter
from mindspore import log as logger

# Need to run all these tests in separate processes since
# the global configuration setting of debug_mode may impact other tests running in parallel.
pytestmark = pytest.mark.forked

DEBUG_MODE = False
SEED_VAL = 0  # seed will be set internally in debug mode, save original seed value to restore.

# tf_file_dataset description:
# test1.data: 10 samples - [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# test2.data: 10 samples - [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# test3.data: 10 samples - [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
# test4.data: 10 samples - [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
# test5.data: 10 samples - [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
TF_FILES = ["../data/dataset/tf_file_dataset/test1.data",
            "../data/dataset/tf_file_dataset/test2.data",
            "../data/dataset/tf_file_dataset/test3.data",
            "../data/dataset/tf_file_dataset/test4.data",
            "../data/dataset/tf_file_dataset/test5.data"]


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
    data = data.rename(input_columns=["image"], output_columns=["image_out"])
    data = data.batch(2)
    num_row = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 2
        assert item["image_out"].shape == (2, 24, 24, 3)
        assert item["attr"].shape == (2, 40)
        num_row += 1
    assert num_row == 2


def test_pipeline_debug_mode_minddata():
    """
    Feature: Pipeline debug mode.
    Description: Test iterator with MindDataset in debug mode.
    Expectation: Successful.
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
    Expectation: Exception raised for unsupported op in debug mode.
    """
    logger.info("test_pipeline_debug_mode_not_support")
    data = ds.NumpySlicesDataset(data=[[0, 1, 2]], column_names=["data"])

    with pytest.raises(RuntimeError) as error_info:
        data.create_tuple_iterator(num_epochs=1, output_numpy=True)
    assert "dataset pipeline" in str(error_info.value)
    assert "Leaf node GeneratorOp is not implemented yet in pull mode." in str(error_info.value)


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


def generator_md():
    """
    Create a dataset with [0, 1, 2, 3, 4]
    """
    for i in range(5):
        yield (np.array([i]),)


# Note: Generator op is not yet supported in pull mode
@pytest.mark.skip(reason="Unsupported in pull mode")
def test_pipeline_debug_mode_skip_take():
    """
    Feature: Pipeline debug mode.
    Description: Test skip op followed by a take op
    Expectation: Output is equal to the expected output
    """
    ds1 = ds.GeneratorDataset(generator_md, ["data"])

    # Here ds1 should be [2, 3, 4]
    ds1 = ds1.skip(2)

    # Here ds1 should be [2, 3]
    ds1 = ds1.take(2)

    buf = []
    for data in ds1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        buf.append(data[0][0])
    assert len(buf) == 2
    assert buf == [2, 3]


def test_pipeline_debug_mode_concat():
    """
    Feature: Pipeline debug mode.
    Description: Test creating tuple iterator with concat.
    Expectation: Successful.
    """
    logger.info("test_pipeline_debug_mode_concat")
    data_dir = "../data/dataset/testImageNetData4/train"
    num_repeat = 2
    data1 = ds.ImageFolderDataset(data_dir, decode=True, num_shards=1, shard_id=0)
    data2 = ds.ImageFolderDataset(data_dir, decode=True, num_shards=1, shard_id=0)
    data3 = ds.ImageFolderDataset(data_dir, decode=True, num_shards=1, shard_id=0)
    data4 = data1.concat(data2)
    data5 = data3 + data4
    data5 = data5.repeat(num_repeat)
    num_epoch = 3
    epoch_count = 0
    sample_row = 21
    sample_count = 0
    for _ in range(num_epoch):
        num_rows = 0
        for item1 in data5.create_tuple_iterator(num_epochs=1):
            assert len(item1) == 2
            assert item1[0].shape == (384, 682, 3)
            num_rows += 1
        epoch_count += 1
        sample_count += num_rows
        assert num_rows == sample_row * num_repeat
    assert epoch_count == num_epoch
    assert sample_count == num_repeat * num_epoch * sample_row


def test_pipeline_debug_mode_tfrecord_rename_zip():
    """
    Feature: Pipeline debug mode.
    Description: Test rename op and zip op followed by repeat
    Expectation: Output is the same as expected output
    """
    tf_data_dir = ["../data/dataset/testTFBert5Rows2/5TFDatas.data"]
    tf_schema_dir = "../data/dataset/testTFBert5Rows2/datasetSchema.json"

    data1 = ds.TFRecordDataset(tf_data_dir, tf_schema_dir, shuffle=False)
    data2 = ds.TFRecordDataset(tf_data_dir, tf_schema_dir, shuffle=False)

    data2 = data2.rename(input_columns=["input_ids", "segment_ids"], output_columns=["masks", "seg_ids"])

    data = ds.zip((data1, data2))
    data = data.repeat(3)

    num_iter = 0
    for _, item in enumerate(data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        logger.info("item[mask] is {}".format(item["masks"]))
        np.testing.assert_equal(item["masks"], item["input_ids"])
        logger.info("item[seg_ids] is {}".format(item["seg_ids"]))
        np.testing.assert_equal(item["segment_ids"], item["seg_ids"])
        # need to consume the data in the buffer
        num_iter += 1
    logger.info("Number of data in data: {}".format(num_iter))
    assert num_iter == 15


def test_pipeline_debug_mode_imagefolder_rename_zip():
    """
    Feature: Pipeline debug mode.
    Description: Test ImageFolderDataset with rename op and zip op
    Expectation: Output is the same as expected output
    """
    # Apply dataset operations
    data1 = ds.ImageFolderDataset("../data/dataset/testPK/data", num_samples=6)
    data2 = ds.ImageFolderDataset("../data/dataset/testPK/data", num_samples=10)

    # Rename dataset2 for no conflict
    data2 = data2.rename(input_columns=["image", "label"], output_columns=["image1", "label1"])
    data2 = data2.skip(4)

    data3 = ds.zip((data1, data2))

    num_iter = 0
    for item in data3.create_dict_iterator(num_epochs=1):  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1
    logger.info("Number of data in data: {}".format(num_iter))
    assert num_iter == 6


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

    # apply dataset operations
    data1 = ds.IMDBDataset("../data/dataset/testIMDBDataset", shuffle=True)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 8
    expect_output = [["train_pos_1.txt", 1], ["train_pos_0.txt", 1], ["train_neg_0.txt", 0], ["test_pos_1.txt", 1], [
        "test_neg_1.txt", 0], ["test_pos_0.txt", 1], ["test_neg_0.txt", 0], ["train_neg_1.txt", 0]]
    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "text" and "label"
        logger.info("text is {}".format(item["text"]))
        assert item["text"] == expect_output[num_iter][0]
        logger.info("label is {}".format(item["label"]))
        assert item["label"] == expect_output[num_iter][1]
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 8


def test_pipeline_debug_mode_tfrecord_shard():
    """
    Feature: Pipeline debug mode
    Description: Test TFRecordDataset shard
    Expectation: The dataset is processed as expected
    """
    logger.info("test_pipeline_debug_mode_tfrecord_shard")

    def get_res(shard_id, num_repeats):
        data1 = ds.TFRecordDataset(TF_FILES[:-1], num_shards=2, shard_id=shard_id, num_samples=3,
                                   shuffle=ds.Shuffle.GLOBAL)
        data1 = data1.repeat(num_repeats)
        res = list()
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(item["scalars"][0])
        return res

    worker1_res = get_res(0, 8)
    worker2_res = get_res(1, 8)
    assert len(worker1_res) == 3 * 8
    assert len(worker1_res) == len(worker2_res)
    for i, _ in enumerate(worker1_res):
        assert worker1_res[i] != worker2_res[i]
    assert set(worker2_res) == set(worker1_res)


def test_pipeline_debug_mode_tfrecord_shard_equal_rows():
    """
    Feature: Pipeline debug mode
    Description: Test TFRecordDataset shard with equal rows in debug mode
    Expectation: The dataset is processed as expected
    """
    logger.info("test_pipeline_debug_mode_tfrecord_shard_equal_rows")

    def get_res(num_shards, shard_id, num_repeats):
        ds1 = ds.TFRecordDataset(TF_FILES[:-1], num_shards=num_shards, shard_id=shard_id, shard_equal_rows=True)
        ds1 = ds1.repeat(num_repeats)
        res = list()
        for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(data["scalars"][0])
        return res

    worker1_res = get_res(3, 0, 2)
    worker2_res = get_res(3, 1, 2)
    worker3_res = get_res(3, 2, 2)
    assert len(worker1_res) == 28
    assert len(worker2_res) == 28
    assert len(worker3_res) == 28

    # Confirm different workers get different results in the same epoch
    for i, _ in enumerate(worker1_res):
        assert worker1_res[i] != worker2_res[i]
        assert worker2_res[i] != worker3_res[i]

    worker4_res = get_res(1, 0, 1)
    assert len(worker4_res) == 40


def test_pipeline_debug_mode_clue_shuffle():
    """
    Feature: Pipeline debug mode
    Description: Test CLUEDataset with Shuffle.GLOBAL parameter versus shuffle op
    Expectation: The dataset is processed as expected
    """
    clue_train_file = '../data/dataset/testCLUE/afqmc/train.json'
    # data1 - CLUEDataset with global shuffle should produce a ShuffleOp over CLUEOp.
    data1 = ds.CLUEDataset(clue_train_file, task='AFQMC', usage='train', shuffle=ds.Shuffle.GLOBAL)
    # data2 - Add explicit shuffle op to pipeline
    data2 = ds.CLUEDataset(clue_train_file, task='AFQMC', usage='train', shuffle=ds.Shuffle.FILES)
    data2 = data2.shuffle(20000)

    for d1, d2 in zip(data1.create_tuple_iterator(num_epochs=1, output_numpy=True),
                      data2.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        for t1, t2 in zip(d1, d2):
            np.testing.assert_array_equal(t1, t2)


if __name__ == '__main__':
    setup_function()
    test_pipeline_debug_mode_tuple()
    test_pipeline_debug_mode_dict()
    test_pipeline_debug_mode_minddata()
    test_pipeline_debug_mode_not_support()
    test_pipeline_debug_mode_map_pyfunc()
    test_pipeline_debug_mode_batch_pyfunc()
    test_pipeline_debug_mode_skip_take()
    test_pipeline_debug_mode_concat()
    test_pipeline_debug_mode_tfrecord_rename_zip()
    test_pipeline_debug_mode_imagefolder_rename_zip()
    test_pipeline_debug_mode_shuffle()
    test_pipeline_debug_mode_map_random()
    test_pipeline_debug_mode_imdb_shuffle()
    test_pipeline_debug_mode_tfrecord_shard()
    test_pipeline_debug_mode_tfrecord_shard_equal_rows()
    test_pipeline_debug_mode_clue_shuffle()
    teardown_function()
