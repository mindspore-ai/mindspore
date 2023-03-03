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
import mindspore.nn as nn
from mindspore.dataset.vision import Inter
from mindspore import log as logger
from mindspore.train import Model

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
TEXTFILE_DATA = "../data/dataset/testTextFileDataset/*"


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


def test_pipeline_debug_mode_numpy_slice_dataset():
    """
    Feature: Pipeline debug mode.
    Description: Test creating tuple iterator with op NumpySlicesDataset in debug mode.
    Expectation: Successful.
    """
    logger.info("test_pipeline_debug_mode_numpy_slice_dataset")

    dataset = ds.NumpySlicesDataset(data=[1, 2, 3], column_names=["column_1"])
    dataset = dataset.map(operations=(lambda x: (x - 1)), input_columns=["column_1"])
    res_exp = np.array([[1], [0], [2]])
    res_actual = []
    row_count = 0
    for item in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 1
        res_actual.append(item)
        row_count += 1
    assert row_count == len(res_exp)
    np.testing.assert_equal(res_actual, res_exp)


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


def test_pipeline_debug_mode_mnist_batch_batch_size_get_batch_num():
    """
    Feature: Pipeline debug mode.
    Description: Test MnistDataset with Batch op using per_batch_map calling get_batch_num()
    Expectation: Output is equal to the expected output
    """
    logger.info("test_pipeline_debug_mode_mnist_batch_batch_size_get_batch_num")

    def add_one(batch_info):
        return batch_info.get_batch_num() + 1

    data = ds.MnistDataset("../data/dataset/testMnistData", num_samples=20)
    data = data.batch(batch_size=add_one, drop_remainder=True)
    num_rows = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_rows += 1
        assert item["label"].shape == (num_rows,)
    assert num_rows == 5


# this generator function yield two columns
# col1d: [0], [1], [2], [3]
# col2d: [[100],[200]], [[101],[201]], [102],[202]], [103],[203]]
def gen_2cols(num):
    for i in range(num):
        yield (np.array([i]), np.array([[i + 100], [i + 200]]))


def test_pipeline_debug_mode_batch_padding():
    """
    Feature: Pipeline debug mode.
    Description: Test padded_batch operation with pad_info
    Expectation: Output is equal to the expected output
    """
    data1 = ds.GeneratorDataset((lambda: gen_2cols(2)), ["col1d", "col2d"])
    # Apply batch padding where input_shape=[x] and output_shape=[y] in which y > x
    data1 = data1.padded_batch(batch_size=2, drop_remainder=False, pad_info={"col2d": ([2, 2], -2), "col1d": ([2], -1)})
    data1 = data1.repeat(2)
    row_count = 0
    for data in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal([[0, -1], [1, -1]], data["col1d"])
        np.testing.assert_array_equal([[[100, -2], [200, -2]], [[101, -2], [201, -2]]], data["col2d"])
        row_count += 1
    assert row_count == 2


def test_pipeline_debug_mode_batch_padding_get_batch_num():
    """
    Feature: Pipeline debug mode.
    Description: Test padded_batch operation with dynamic batch_size that uses get_batch_num()
    Expectation: Output is equal to the expected output
    """

    def add_one(batch_info):
        return batch_info.get_batch_num() + 1

    data1 = ds.GeneratorDataset((lambda: gen_2cols(3)), ["col1d", "col2d"],
                                python_multiprocessing=False)
    # Apply batch padding with dynamic batch_size
    data1 = data1.padded_batch(batch_size=add_one, drop_remainder=False,
                               pad_info={"col2d": ([2, 2], -2), "col1d": ([2], -1)},
                               num_parallel_workers=1)
    data1 = data1.repeat(2)

    exp1 = [[[0, -1]],
            [[1, -1], [2, -1]],
            [[0, -1]],
            [[1, -1], [2, -1]]]
    exp2 = [[[[100, -2], [200, -2]]],
            [[[101, -2], [201, -2]], [[102, -2], [202, -2]]],
            [[[100, -2], [200, -2]]],
            [[[101, -2], [201, -2]], [[102, -2], [202, -2]]]]
    i = 0
    for data in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(exp1[i], data["col1d"])
        np.testing.assert_array_equal(exp2[i], data["col2d"])
        i += 1
    assert i == 4


def generator_md():
    """
    Create a dataset with [0-9]
    """
    for i in range(10):
        yield (np.array([i]),)


def test_pipeline_debug_mode_generator_pipeline():
    """
    Feature: Pipeline debug mode.
    Description: Test generator-skip-take-shuffle-batch pipeline
    Expectation: Output is equal to the expected output
    """
    logger.info("test_pipeline_debug_mode_generator_pipeline")
    ds1 = ds.GeneratorDataset(generator_md, ["data"])

    # Here ds1 should be [2, 3, 4, 5, 6, 7, 8, 9]
    ds1 = ds1.skip(2)

    # Here ds1 should be [2, 3, 4, 5, 6, 7, 8]
    ds1 = ds1.take(7)

    # do shuffle followed by batch
    # Note: Since the internal seed is set in debug mode, the consistency of results after ShuffleOp can be ensured.
    ds1 = ds1.shuffle(5)
    ds1 = ds1.batch(3, drop_remainder=True)

    buf = []
    for data in ds1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        buf.append(data[0])
    assert len(buf) == 2
    out_expect = [[[5], [4], [2]], [[7], [8], [3]]]
    np.testing.assert_array_equal(buf, out_expect)


def test_pipeline_debug_mode_generator_repeat():
    """
    Feature: Pipeline debug mode
    Description: Test generator op followed by a repeat op
    Expectation: Output is equal to the expected output
    """
    logger.info("test_pipeline_debug_mode_generator_repeat")
    num_rows = 5
    num_repeates = 2
    ds1 = ds.GeneratorDataset(generator_md, ["data"], num_samples=num_rows)
    ds1 = ds1.repeat(num_repeates)

    buf = []
    for data in ds1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        buf.append(data[0][0])
    assert len(buf) == num_rows * num_repeates
    out_expect = list(range(num_rows)) * num_repeates
    assert buf == out_expect


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
    Expectation: Successful.
    """
    logger.info("test_pipeline_debug_mode_shuffle")

    buffer_size = 5
    data = ds.TextFileDataset(TEXTFILE_DATA, shuffle=False)
    # Note: Since the internal seed is set in debug mode, the consistency of results after ShuffleOp can be ensured.
    data = data.shuffle(buffer_size=buffer_size)
    out_expect = ["End of file.", "Be happy every day.", "This is a text file.", "Good luck to everyone.",
                  "Another file."]
    num_rows = 0
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert item["text"] == out_expect[num_rows]
        num_rows += 1
    assert num_rows == 5


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


def run_celeba_pyop_pipeline(python_multiprocessing=False):
    """ Create and execute simple CelebADataset pipeline with Python implemented ops. """

    # Create CelebADataset pipeline with Python implemented ops
    data1 = ds.CelebADataset("../data/dataset/testCelebAData/")
    data1 = data1.map(operations=[vision.Decode(to_pil=True)], input_columns=["image"])
    data1 = data1.map(operations=[vision.Resize((20, 30))], input_columns=["image"],
                      python_multiprocessing=python_multiprocessing)
    data1 = data1.map(operations=[lambda x: x], input_columns=["image"],
                      python_multiprocessing=python_multiprocessing)
    data1 = data1.batch(batch_size=2, python_multiprocessing=python_multiprocessing)

    # Iterate over data pipeline
    row_count = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 2
        assert item[0].shape == (2, 20, 30, 3)
        assert item[1].shape == (2, 40)
        row_count += 1
    assert row_count == 2


def test_pipeline_debug_mode_celeba_pyop():
    """
    Feature: Debug Mode
    Description: Test Debug Mode enabled with CelebADataset with Python implemented ops
    Expectation: Sanity check of data pipeline is done. Output is equal to the expected output
    """
    # Create and execute data pipeline
    run_celeba_pyop_pipeline()


def test_pipeline_debug_mode_celeba_pyop_mp():
    """
    Feature: Debug Mode
    Description: Test Debug Mode enabled with CelebADataset with python_multiprocessing=True
    Expectation: Sanity check of data pipeline is done. Output is equal to the expected output
    """
    # Reduce memory required by disabling the shared memory optimization
    mem_original = ds.config.get_enable_shared_mem()
    ds.config.set_enable_shared_mem(False)

    # Create and execute data pipeline
    run_celeba_pyop_pipeline(python_multiprocessing=True)

    # Restore configuration
    ds.config.set_enable_shared_mem(mem_original)


def test_pipeline_debug_mode_im_per_batch_map():
    """
    Feature: Debug Mode
    Description: Test Debug Mode enabled with Batch op using per_batch_map parameter
    Expectation: Output is equal to the expected output
    """

    def split_imgs_and_labels(imgs, labels, batch_info):
        """split data into labels and images"""
        ret_imgs = []
        ret_labels = []

        for i, image in enumerate(imgs):
            ret_imgs.append(image)
            ret_labels.append(labels[i])
        return np.array(ret_imgs), np.array(ret_labels)

    data1 = ds.ImageFolderDataset("../data/dataset/testImageNetData4/train", decode=True)
    batch_size = 1
    data1 = data1.batch(batch_size=batch_size, drop_remainder=True,
                        input_columns=["image", "label"],
                        per_batch_map=split_imgs_and_labels)

    row_count = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 2
        assert item["image"].shape == (batch_size, 384, 682, 3)
        row_count += 1
    assert row_count == 7


def test_pipeline_debug_mode_im_per_batch_map_resize():
    """
    Feature: Debug Mode
    Description: Test Debug Mode enabled with Batch op plus per_batch_map that resizes and dynamic batch_size
    Expectation: Output is equal to the expected output
    """

    # Fake resize image according to its batch number. e.g. if it's 5-th batch, resize to (5^2, 5^2) = (25, 25)
    def np_pseudo_resize(col, batch_info):
        s = (batch_info.get_batch_num() + 1) ** 2
        return ([np.copy(c[0:s, 0:s, :]) for c in col],)

    def add_one(batch_info):
        return batch_info.get_batch_num() + 1

    data1 = ds.ImageFolderDataset("../data/dataset/testPK/data/", decode=True)
    # Set batch_size to callable function that dynamics updates the batch_size
    # Set per_batch_map to callable function that resizes the input image
    data1 = data1.batch(batch_size=add_one, drop_remainder=True,
                        input_columns=["image"], per_batch_map=np_pseudo_resize)
    # i-th batch has shape (i, i^2, i^2, 3)
    i = 1
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert item["image"].shape == (i, i ** 2, i ** 2, 3)
        i += 1
    # Note: There are 9-1=8 rows
    assert i == 9


def run_cifar100_per_batch_map_pipeline(python_multiprocessing=False, num_parallel_workers=1):
    """ Create and execute Cifar100 dataset pipline with Batch op using per_batch_map """
    # Define dataset pipeline
    num_samples = 300
    cifar100_ds = ds.Cifar100Dataset("../data/dataset/testCifar100Data", num_samples=num_samples)
    cifar100_ds = cifar100_ds.map(operations=[vision.ToType(np.int32)], input_columns="fine_label")
    cifar100_ds = cifar100_ds.map(operations=[lambda z: z], input_columns="image")

    # Callable function to delete 3rd column
    def del_column(col1, col2, col3, batch_info):
        return (col1, col2,)

    # Apply Dataset Ops
    buffer_size = 10000
    batch_size = 32
    repeat_size = 3
    cifar100_ds = cifar100_ds.shuffle(buffer_size=buffer_size)
    # Note: Test repeat before batch
    cifar100_ds = cifar100_ds.repeat(repeat_size)
    cifar100_ds = cifar100_ds.batch(batch_size, per_batch_map=del_column,
                                    input_columns=['image', 'fine_label', 'coarse_label'],
                                    output_columns=['image', 'label'], drop_remainder=True,
                                    num_parallel_workers=num_parallel_workers,
                                    python_multiprocessing=python_multiprocessing)

    # Iterate over data pipeline
    row_count = 0
    for item in cifar100_ds.create_dict_iterator(num_epochs=1):
        assert len(item) == 2
        assert item["image"].shape == (batch_size, 32, 32, 3)
        row_count += 1
    assert row_count == 28


def test_pipeline_debug_mode_cifar100_per_batch_map():
    """
    Feature: Debug Mode
    Description: Test Debug Mode enabled with Cifar100Dataset with Batch op using per_batch_map
    Expectation: Sanity check of data pipeline is done. Output is equal to the expected output
    """
    # Create and execute data pipeline
    # Note: In Debug Mode, using num_parallel_workers > 1 is ignored
    run_cifar100_per_batch_map_pipeline(python_multiprocessing=False, num_parallel_workers=3)


def test_pipeline_debug_mode_cifar100_per_batch_map_mp():
    """
    Feature: Debug Mode
    Description: Test Debug Mode enabled with Cifar100Dataset with Batch op using per_batch_map and
        python_multiprocessing=True
    Expectation: Sanity check of data pipeline is done. Output is equal to the expected output
    """
    # Reduce memory required by disabling the shared memory optimization
    mem_original = ds.config.get_enable_shared_mem()
    ds.config.set_enable_shared_mem(False)

    # Create and execute data pipeline
    # Note: In Debug Mode, using num_parallel_workers > 1 is ignored
    run_cifar100_per_batch_map_pipeline(python_multiprocessing=True, num_parallel_workers=5)

    # Restore configuration
    ds.config.set_enable_shared_mem(mem_original)


def test_pipeline_debug_mode_batch_map_get_epoch_batch_num():
    """
    Feature: Debug Mode
    Description: Test Debug Mode with Batch op with per_batch_map calling get_epoch_num(), get_batch_num()
    Expectation: Output is equal to the expected output
    """

    def check_res(arr1, arr2):
        assert len(arr1) == len(arr2)
        for ind, _ in enumerate(arr1):
            assert np.array_equal(arr1[ind], np.array(arr2[ind]))

    def gen(num):
        for i in range(num):
            yield (np.array([i]),)

    def invert_sign_per_epoch(col_list, batch_info):
        return ([np.copy(((-1) ** batch_info.get_epoch_num()) * arr) for arr in col_list],)

    def invert_sign_per_batch(col_list, batch_info):
        return ([np.copy(((-1) ** batch_info.get_batch_num()) * arr) for arr in col_list],)

    def batch_map_config(num, r, batch_size, func, res):
        data1 = ds.GeneratorDataset((lambda: gen(num)), ["num"],
                                    python_multiprocessing=False)
        data1 = data1.batch(batch_size=batch_size, input_columns=["num"], per_batch_map=func,
                            python_multiprocessing=False, num_parallel_workers=1)
        data1 = data1.repeat(r)
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(item["num"])

    tst1, tst2, = [], []
    batch_map_config(4, 2, 2, invert_sign_per_epoch, tst1)
    # Note: There is only one epoch; get_epoch_num() returns 0.
    check_res(tst1, [[[0], [1]], [[2], [3]], [[0], [1]], [[2], [3]]])

    # For each batch, the sign of a row is changed.
    batch_map_config(4, 2, 2, invert_sign_per_batch, tst2)
    check_res(tst2, [[[0], [1]], [[-2], [-3]], [[0], [1]], [[-2], [-3]]])


def test_pipeline_debug_mode_batch_map_get_epoch_num():
    """
    Feature: Dataset Debug Mode
    Description: Test basic map Batch op with per_batch_map function calling get_epoch_num()
    Expectation: Output is equal to the expected output
    """

    def check_res(arr1, arr2):
        assert len(arr1) == len(arr2)
        for ind, _ in enumerate(arr1):
            assert np.array_equal(arr1[ind], np.array(arr2[ind]))

    def gen(num):
        for i in range(num):
            yield (np.array([i]),)

    def invert_sign_per_epoch(col_list, batch_info):
        return ([np.copy(((-1) ** batch_info.get_epoch_num()) * arr) for arr in col_list],)

    def batch_map_config(num_samples, num_repeat, batch_size, num_epoch, func, res):
        data1 = ds.GeneratorDataset((lambda: gen(num_samples)), ["num"],
                                    python_multiprocessing=False)
        data1 = data1.batch(batch_size=batch_size, input_columns=["num"], per_batch_map=func,
                            python_multiprocessing=False, num_parallel_workers=1)
        data1 = data1.repeat(num_repeat)

        # Iterate over data pipeline
        iter1 = data1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)
        epoch_count = 0
        sample_count = 0
        for _ in range(num_epoch):
            row_count = 0
            for item in iter1:
                res.append(item["num"])
                row_count += 1
            assert row_count == int(num_samples * num_repeat / batch_size)
            epoch_count += 1
            sample_count += row_count
        assert epoch_count == num_epoch
        assert sample_count == int(num_samples * num_repeat / batch_size) * num_epoch

    res1, res2, res3, res4, = [], [], [], []

    # Test repeat=1, num_epochs=1
    batch_map_config(4, 1, 2, 1, invert_sign_per_epoch, res1)
    check_res(res1, [[[0], [1]], [[2], [3]]])

    # Test repeat=2, num_epochs=1
    batch_map_config(4, 2, 2, 1, invert_sign_per_epoch, res2)
    check_res(res2, [[[0], [1]], [[2], [3]], [[0], [1]], [[2], [3]]])

    # Test repeat=1, num_epochs=2
    batch_map_config(4, 1, 2, 2, invert_sign_per_epoch, res3)
    check_res(res3, [[[0], [1]], [[2], [3]], [[0], [-1]], [[-2], [-3]]])

    # Test repeat=2, num_epochs=2
    batch_map_config(4, 2, 2, 2, invert_sign_per_epoch, res4)
    check_res(res4, [[[0], [1]], [[2], [3]], [[0], [1]], [[2], [3]],
                     [[0], [-1]], [[-2], [-3]], [[0], [-1]], [[-2], [-3]]])


def test_pipeline_debug_mode_dataset_sink_not_support():
    """
    Feature: Dataset Debug Mode
    Description: Test dataset sink mode with debug mode enabled.
    Expectation: Raise ValueError and give proper log.
    """
    dataset = ds.Cifar100Dataset("../data/dataset/testCifar100Data", num_samples=100)
    def create_model():
        class Net(nn.Cell):
            def construct(self, x):
                return x
        net = Net()
        return Model(net)
    model = create_model()
    with pytest.raises(ValueError) as error_info:
        model.train(2, dataset, dataset_sink_mode=True)
    assert "Dataset sink mode is not supported when dataset pipeline debug mode is on. "\
           "Please manually turn off sink mode" in str(error_info.value)


if __name__ == '__main__':
    setup_function()
    test_pipeline_debug_mode_tuple()
    test_pipeline_debug_mode_dict()
    test_pipeline_debug_mode_minddata()
    test_pipeline_debug_mode_numpy_slice_dataset()
    test_pipeline_debug_mode_map_pyfunc()
    test_pipeline_debug_mode_mnist_batch_batch_size_get_batch_num()
    test_pipeline_debug_mode_batch_padding()
    test_pipeline_debug_mode_batch_padding_get_batch_num()
    test_pipeline_debug_mode_generator_pipeline()
    test_pipeline_debug_mode_generator_repeat()
    test_pipeline_debug_mode_concat()
    test_pipeline_debug_mode_tfrecord_rename_zip()
    test_pipeline_debug_mode_imagefolder_rename_zip()
    test_pipeline_debug_mode_map_random()
    test_pipeline_debug_mode_shuffle()
    test_pipeline_debug_mode_imdb_shuffle()
    test_pipeline_debug_mode_tfrecord_shard()
    test_pipeline_debug_mode_tfrecord_shard_equal_rows()
    test_pipeline_debug_mode_clue_shuffle()
    test_pipeline_debug_mode_celeba_pyop()
    test_pipeline_debug_mode_celeba_pyop_mp()
    test_pipeline_debug_mode_im_per_batch_map()
    test_pipeline_debug_mode_im_per_batch_map_resize()
    test_pipeline_debug_mode_cifar100_per_batch_map()
    test_pipeline_debug_mode_cifar100_per_batch_map_mp()
    test_pipeline_debug_mode_batch_map_get_epoch_batch_num()
    test_pipeline_debug_mode_batch_map_get_epoch_num()
    test_pipeline_debug_mode_dataset_sink_not_support()
    teardown_function()
