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
Test Cifar10 and Cifar100 dataset operations in debug mode
"""
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
import mindspore.dataset as ds
from mindspore import log as logger

pytestmark = pytest.mark.forked

DATA_DIR_10 = "../data/dataset/testCifar10Data"
DATA_DIR_100 = "../data/dataset/testCifar100Data"
NO_BIN_DIR = "../data/dataset/testMnistData"
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


def load_cifar(path, kind="cifar10"):
    """
    load Cifar10/100 data
    """
    raw = np.empty(0, dtype=np.uint8)
    for file_name in os.listdir(path):
        if file_name.endswith(".bin"):
            with open(os.path.join(path, file_name), mode='rb') as file:
                raw = np.append(raw, np.fromfile(file, dtype=np.uint8), axis=0)
    if kind == "cifar10":
        raw = raw.reshape(-1, 3073)
        labels = raw[:, 0]
        images = raw[:, 1:]
    elif kind == "cifar100":
        raw = raw.reshape(-1, 3074)
        labels = raw[:, :2]
        images = raw[:, 2:]
    else:
        raise ValueError("Invalid parameter value")
    images = images.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)
    return images, labels


def visualize_dataset(images, labels):
    """
    Helper function to visualize the dataset samples
    """
    num_samples = len(images)
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i])
        plt.title(labels[i])
    plt.show()


### Testcases for Cifar10Dataset Op ###


def test_cifar10_content_check():
    """
    Feature: Pipeline debug mode with Cifar10Dataset
    Description: Test Cifar10Dataset with content check on image readings in pull mode
    Expectation: The dataset is processed as expected
    """
    logger.info("Test debug mode Cifar10Dataset Op with content check")
    data1 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=100, shuffle=False)
    images, labels = load_cifar(DATA_DIR_10)
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label"
    for i, d in enumerate(data1.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(d["image"], images[i])
        np.testing.assert_array_equal(d["label"], labels[i])
        num_iter += 1
    assert num_iter == 100


def test_cifar10_basic():
    """
    Feature: Cifar10Dataset
    Description: Test Cifar10Dataset with some basic arguments and methods
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Cifar10Dataset Op")

    # case 0: test loading the whole dataset
    data0 = ds.Cifar10Dataset(DATA_DIR_10)
    num_iter0 = 0
    for _ in data0.create_dict_iterator(num_epochs=1):
        num_iter0 += 1
    assert num_iter0 == 10000

    # case 1: test num_samples
    data1 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=100)
    num_iter1 = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter1 += 1
    assert num_iter1 == 100

    # case 2: test batch with drop_remainder=False
    data2 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=100)
    assert data2.get_dataset_size() == 100
    assert data2.get_batch_size() == 1
    data2 = data2.batch(batch_size=7)  # drop_remainder is default to be False
    assert data2.get_dataset_size() == 15
    assert data2.get_batch_size() == 7
    num_iter2 = 0
    for _ in data2.create_dict_iterator(num_epochs=1):
        num_iter2 += 1
    assert num_iter2 == 15

    # case 3: test batch with drop_remainder=True
    data3 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=100)
    assert data3.get_dataset_size() == 100
    assert data3.get_batch_size() == 1
    data3 = data3.batch(batch_size=7, drop_remainder=True)  # the rest of incomplete batch will be dropped
    assert data3.get_dataset_size() == 14
    assert data3.get_batch_size() == 7
    num_iter3 = 0
    for _ in data3.create_dict_iterator(num_epochs=1):
        num_iter3 += 1
    assert num_iter3 == 14


def test_cifar10_pk_sampler():
    """
    Feature: Pipeline debug mode with Cifar10Dataset
    Description: Test Cifar10Dataset with PKSampler in debug mode
    Expectation: The dataset is processed as expected
    """
    logger.info("Test debug mode Cifar10Dataset Op with PKSampler")
    golden = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
              5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]
    sampler = ds.PKSampler(3)
    data = ds.Cifar10Dataset(DATA_DIR_10, sampler=sampler)
    num_iter = 0
    label_list = []
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        label_list.append(item["label"])
        num_iter += 1
    np.testing.assert_array_equal(golden, label_list)
    assert num_iter == 30


def test_cifar10_sequential_sampler():
    """
    Feature: Pipeline debug mode with Cifar10Dataset
    Description: Test Cifar10Dataset with SequentialSampler in debug mode
    Expectation: The dataset is processed as expected
    """
    logger.info("Test debug mode Cifar10Dataset Op with SequentialSampler")
    num_samples = 30
    sampler = ds.SequentialSampler(num_samples=num_samples)
    data1 = ds.Cifar10Dataset(DATA_DIR_10, sampler=sampler)
    data2 = ds.Cifar10Dataset(DATA_DIR_10, shuffle=False, num_samples=num_samples)
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_equal(item1["label"], item2["label"])
        num_iter += 1
    assert num_iter == num_samples


def test_cifar10_exception():
    """
    Feature: Pipeline debug mode with Cifar10Dataset
    Description: Test error cases Cifar10Dataset in debug mode
    Expectation: Throw correct error as expected
    """
    logger.info("Test error cases for Cifar10Dataset in debug mode")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.Cifar10Dataset(DATA_DIR_10, shuffle=False, sampler=ds.PKSampler(3))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.Cifar10Dataset(DATA_DIR_10, sampler=ds.PKSampler(3), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.Cifar10Dataset(DATA_DIR_10, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.Cifar10Dataset(DATA_DIR_10, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.Cifar10Dataset(DATA_DIR_10, num_shards=2, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.Cifar10Dataset(DATA_DIR_10, num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.Cifar10Dataset(DATA_DIR_10, shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.Cifar10Dataset(DATA_DIR_10, shuffle=False, num_parallel_workers=256)

    error_msg_7 = r"cifar\(.bin\) files are missing"
    with pytest.raises(RuntimeError, match=error_msg_7):
        ds1 = ds.Cifar10Dataset(NO_BIN_DIR)
        for _ in ds1.__iter__():
            pass


def test_cifar10_visualize(plot=False):
    """
    Feature: Pipeline debug mode with Cifar10Dataset
    Description: Test Cifar10Dataset visualization results in debug mode
    Expectation: Results are presented as expected in debug mode
    """
    logger.info("Test debug mode Cifar10Dataset visualization")

    data1 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=10, shuffle=False)
    num_iter = 0
    image_list, label_list = [], []
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item["image"]
        label = item["label"]
        image_list.append(image)
        label_list.append("label {}".format(label))
        assert isinstance(image, np.ndarray)
        assert image.shape == (32, 32, 3)
        assert image.dtype == np.uint8
        assert label.dtype == np.uint32
        num_iter += 1
    assert num_iter == 10
    if plot:
        visualize_dataset(image_list, label_list)


### Testcases for Cifar100Dataset Op ###

def test_cifar100_content_check():
    """
    Feature: Pipeline debug mode with Cifar10Dataset
    Description: Test Cifar100Dataset image readings with content check in debug mode
    Expectation: The dataset is processed as expected
    """
    logger.info("Test debug mode Cifar100Dataset with content check")
    data1 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100, shuffle=False)
    images, labels = load_cifar(DATA_DIR_100, kind="cifar100")
    num_iter = 0
    # in this example, each dictionary has keys "image", "coarse_label" and "fine_image"
    for i, d in enumerate(data1.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(d["image"], images[i])
        np.testing.assert_array_equal(d["coarse_label"], labels[i][0])
        np.testing.assert_array_equal(d["fine_label"], labels[i][1])
        num_iter += 1
    assert num_iter == 100


def test_cifar100_basic():
    """
    Feature: Pipeline debug mode with Cifar10Dataset
    Description: Test Cifar100Dataset basic arguments and features in debug mode
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Cifar100Dataset basic in debug mode")

    # case 1: test num_samples
    data1 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100)
    num_iter1 = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter1 += 1
    assert num_iter1 == 100

    # case 2: test batch with drop_remainder=True
    data2 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100)
    data2 = data2.batch(batch_size=3, drop_remainder=True)
    assert data2.get_dataset_size() == 33
    assert data2.get_batch_size() == 3
    num_iter2 = 0
    for _ in data2.create_dict_iterator(num_epochs=1):
        num_iter2 += 1
    assert num_iter2 == 33

    # case 3: test batch with drop_remainder=False
    data3 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100)
    assert data3.get_dataset_size() == 100
    assert data3.get_batch_size() == 1
    data3 = data3.batch(batch_size=3)
    assert data3.get_dataset_size() == 34
    assert data3.get_batch_size() == 3
    num_iter3 = 0
    for _ in data3.create_dict_iterator(num_epochs=1):
        num_iter3 += 1
    assert num_iter3 == 34


def test_cifar100_pk_sampler():
    """
    Feature: Pipeline debug mode with Cifar10Dataset
    Description: Test Cifar100Dataset with PKSampler in debug mode
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Cifar100Dataset with PKSampler in deubg mode")
    golden = [i for i in range(20)]
    sampler = ds.PKSampler(1)
    data = ds.Cifar100Dataset(DATA_DIR_100, sampler=sampler)
    num_iter = 0
    label_list = []
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        label_list.append(item["coarse_label"])
        num_iter += 1
    np.testing.assert_array_equal(golden, label_list)
    assert num_iter == 20


def test_cifar100_exception():
    """
    Feature: Pipeline debug mode with Cifar10Dataset
    Description: Test error cases for Cifar100Dataset in debug mode
    Expectation: Throw correct error as expected
    """
    logger.info("Test error cases for Cifar100Dataset in debug mode")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.Cifar100Dataset(DATA_DIR_100, shuffle=False, sampler=ds.PKSampler(3))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.Cifar100Dataset(DATA_DIR_100, sampler=ds.PKSampler(3), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.Cifar100Dataset(DATA_DIR_100, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.Cifar100Dataset(DATA_DIR_100, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.Cifar100Dataset(DATA_DIR_100, num_shards=2, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.Cifar10Dataset(DATA_DIR_100, num_shards=2, shard_id=5)

    error_msg_7 = r"cifar\(.bin\) files are missing"
    with pytest.raises(RuntimeError, match=error_msg_7):
        ds1 = ds.Cifar100Dataset(NO_BIN_DIR)
        for _ in ds1.__iter__():
            pass


def test_cifar100_visualize(plot=False):
    """
    Feature: Pipeline debug mode with Cifar10Dataset
    Description: Test Cifar100Dataset visualization results in debug mode
    Expectation: Results are presented as expected
    """
    logger.info("Test Cifar100Dataset visualization in debug mode")

    data1 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=10, shuffle=False)
    num_iter = 0
    image_list, label_list = [], []
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item["image"]
        coarse_label = item["coarse_label"]
        fine_label = item["fine_label"]
        image_list.append(image)
        label_list.append("coarse_label {}\nfine_label {}".format(coarse_label, fine_label))
        assert isinstance(image, np.ndarray)
        assert image.shape == (32, 32, 3)
        assert image.dtype == np.uint8
        assert coarse_label.dtype == np.uint32
        assert fine_label.dtype == np.uint32
        num_iter += 1
    assert num_iter == 10
    if plot:
        visualize_dataset(image_list, label_list)


def test_cifar_usage():
    """
    Feature: Pipeline debug mode with Cifar10Dataset
    Description: Test Cifar100Dataset usage flag in debug mode
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Cifar100Dataset usage flag in defbug mode")

    # flag, if True, test cifar10 else test cifar100
    def test_config(usage, flag=True, cifar_path=None):
        if cifar_path is None:
            cifar_path = DATA_DIR_10 if flag else DATA_DIR_100
        try:
            data = ds.Cifar10Dataset(cifar_path, usage=usage) if flag else ds.Cifar100Dataset(cifar_path, usage=usage)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    # test the usage of CIFAR100
    assert test_config("train") == 10000
    assert test_config("all") == 10000
    assert "usage is not within the valid set of ['train', 'test', 'all']" in test_config("invalid")
    assert "Argument usage with value ['list'] is not of type [<class 'str'>]" in test_config(["list"])
    assert "Cifar10Dataset API can't read the data file (interface mismatch or no data found)" in test_config("test")

    # test the usage of CIFAR10
    assert test_config("test", False) == 10000
    assert test_config("all", False) == 10000
    assert "Cifar100Dataset API can't read the data file" in test_config("train", False)
    assert "usage is not within the valid set of ['train', 'test', 'all']" in test_config("invalid", False)

    # change this directory to the folder that contains all cifar10 files
    all_cifar10 = None
    if all_cifar10 is not None:
        assert test_config("train", True, all_cifar10) == 50000
        assert test_config("test", True, all_cifar10) == 10000
        assert test_config("all", True, all_cifar10) == 60000
        assert ds.Cifar10Dataset(all_cifar10, usage="train").get_dataset_size() == 50000
        assert ds.Cifar10Dataset(all_cifar10, usage="test").get_dataset_size() == 10000
        assert ds.Cifar10Dataset(all_cifar10, usage="all").get_dataset_size() == 60000

    # change this directory to the folder that contains all cifar100 files
    all_cifar100 = None
    if all_cifar100 is not None:
        assert test_config("train", False, all_cifar100) == 50000
        assert test_config("test", False, all_cifar100) == 10000
        assert test_config("all", False, all_cifar100) == 60000
        assert ds.Cifar100Dataset(all_cifar100, usage="train").get_dataset_size() == 50000
        assert ds.Cifar100Dataset(all_cifar100, usage="test").get_dataset_size() == 10000
        assert ds.Cifar100Dataset(all_cifar100, usage="all").get_dataset_size() == 60000


def test_cifar_exception_file_path():
    """
    Feature: Pipeline debug mode with Cifar10Dataset
    Description: Test Cifar10Dataset and Cifar100Dataset with invalid file path in debug mode
    Expectation: Error is raised as expected
    """

    def exception_func(item):
        raise Exception("Error occur!")

    with pytest.raises(RuntimeError) as error_info:
        data = ds.Cifar10Dataset(DATA_DIR_10)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
    assert "map operation: [PyFunc] failed. The corresponding data file is" in str(error_info.value)

    with pytest.raises(RuntimeError) as error_info:
        data = ds.Cifar10Dataset(DATA_DIR_10)
        data = data.map(operations=exception_func, input_columns=["label"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
    assert "map operation: [PyFunc] failed. The corresponding data file is" in str(error_info.value)

    with pytest.raises(RuntimeError) as error_info:
        data = ds.Cifar100Dataset(DATA_DIR_100)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
    assert "map operation: [PyFunc] failed. The corresponding data file is" in str(error_info.value)

    with pytest.raises(RuntimeError) as error_info:
        data = ds.Cifar100Dataset(DATA_DIR_100)
        data = data.map(operations=exception_func, input_columns=["coarse_label"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
    assert "map operation: [PyFunc] failed. The corresponding data file is" in str(error_info.value)

    with pytest.raises(RuntimeError) as error_info:
        data = ds.Cifar100Dataset(DATA_DIR_100)
        data = data.map(operations=exception_func, input_columns=["fine_label"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
        assert False
    assert "map operation: [PyFunc] failed. The corresponding data file is" in str(error_info.value)


def test_cifar10_pk_sampler_get_dataset_size():
    """
    Feature: Pipeline debug mode with Cifar10Dataset
    Description: Test Cifar10Dataset get_dataset_size in debug mode
    Expectation: The dataset is processed as expected
    """
    sampler = ds.PKSampler(3)
    data = ds.Cifar10Dataset(DATA_DIR_10, sampler=sampler)
    num_iter = 0
    ds_sz = data.get_dataset_size()
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1

    assert ds_sz == num_iter == 30


def test_cifar10_with_chained_sampler_get_dataset_size():
    """
    Feature: Cifar10Dataset
    Description: Test Cifar10Dataset with PKSampler chained with a SequentialSampler and get_dataset_size
    Expectation: The dataset is processed as expected
    """
    sampler = ds.SequentialSampler(start_index=0, num_samples=5)
    child_sampler = ds.PKSampler(4)
    sampler.add_child(child_sampler)
    data = ds.Cifar10Dataset(DATA_DIR_10, sampler=sampler)
    num_iter = 0
    ds_sz = data.get_dataset_size()
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    assert ds_sz == num_iter == 5


def test_cifar100ops():
    """
    Feature: Pipeline debug mode with Cifar10Dataset
    Description: Test Cifar100Dataset with take and skip operations in debug mode
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Cifar100Dataset operations in debug mode")

    # case 1: test num_samples
    data1 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100)
    num_iter1 = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter1 += 1
    assert num_iter1 == 100

    # take 30
    num_iter2 = 0
    data2 = data1.take(30)
    for _ in data2.create_dict_iterator(num_epochs=1):
        num_iter2 += 1
    assert num_iter2 == 30

    # take default 0
    data3 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100)
    num_iter3 = 0
    data3 = data3.take()
    for _ in data3.create_dict_iterator(num_epochs=1):
        num_iter3 += 1
    assert num_iter3 == 100

    # take more than dataset size
    data4 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100)
    num_iter4 = 0
    data4 = data4.take(1000)
    for _ in data4.create_dict_iterator(num_epochs=1):
        num_iter4 += 1
    assert num_iter4 == 100

    # take -5
    data5 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100)
    with pytest.raises(ValueError) as error_info:
        data5 = data5.take(-5)
        for _ in data4.create_dict_iterator(num_epochs=1):
            pass
    assert "count should be either -1 or within the required interval" in str(error_info.value)

    # skip 0
    data6 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100)
    num_iter6 = 0
    data6 = data6.skip(0)
    for _ in data6.create_dict_iterator(num_epochs=1):
        num_iter6 += 1
    assert num_iter6 == 100

    # skip more than dataset size
    data7 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100)
    num_iter7 = 0
    data7 = data7.skip(1000)
    for _ in data7.create_dict_iterator(num_epochs=1):
        num_iter7 += 1
    assert num_iter7 == 0

    # skip -5
    data8 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100)
    with pytest.raises(ValueError) as error_info:
        data8 = data8.skip(-5)
        for _ in data8.create_dict_iterator(num_epochs=1):
            pass
    assert "Input count is not within the required interval of" in str(error_info.value)


### Focused debug mode testcases with Cifar10Dataset ###

def test_pipeline_debug_mode_cifar10_rename_zip(plot=False):
    """
    Feature: Pipeline debug mode.
    Description: Test Cifar10Dataset with rename op and zip op for 2 datasets
    Expectation: Output is the same as expected output
    """

    def test_config(num_samples1, num_samples2, plot):
        # Apply dataset operations
        data1 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=num_samples1)
        data2 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=num_samples2)

        # Rename dataset2 for no conflict
        data2 = data2.rename(input_columns=["image", "label"], output_columns=["image2", "label2"])

        data3 = ds.zip((data1, data2))

        num_iter = 0
        image_list, image_list2, label_list, label_list2 = [], [], [], []
        for item in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
            image = item["image"]
            label = item["label"]
            image_list.append(image)
            label_list.append("label {}".format(label))
            assert isinstance(image, np.ndarray)
            assert image.shape == (32, 32, 3)
            assert image.dtype == np.uint8
            assert label.dtype == np.uint32

            image2 = item["image2"]
            label2 = item["label2"]
            image_list2.append(image2)
            label_list2.append("label {}".format(label2))
            assert isinstance(image2, np.ndarray)
            assert image2.shape == (32, 32, 3)
            assert image2.dtype == np.uint8
            assert label2.dtype == np.uint32

            assert label == label2
            np.testing.assert_equal(image, image2)

            num_iter += 1
        assert num_iter == min(num_samples1, num_samples2)

        if plot:
            visualize_dataset(image_list, label_list)
            visualize_dataset(image_list2, label_list2)

    # Test zip with sample number of samples for both datasets
    test_config(6, 6, plot)
    # Test zip with more samples for 2nd dataset (child 1)
    test_config(4, 7, plot)
    # Test zip with more samples for 1st dataset (child 0)
    test_config(13, 8, plot)


if __name__ == '__main__':
    setup_function()
    test_cifar10_content_check()
    test_cifar10_basic()
    test_cifar10_pk_sampler()
    test_cifar10_sequential_sampler()
    test_cifar10_exception()
    test_cifar10_visualize(plot=False)
    test_cifar100_content_check()
    test_cifar100_basic()
    test_cifar100_pk_sampler()
    test_cifar100_exception()
    test_cifar100_visualize(plot=False)
    test_cifar_usage()
    test_cifar_exception_file_path()
    test_cifar10_with_chained_sampler_get_dataset_size()
    test_cifar10_pk_sampler_get_dataset_size()
    test_cifar100ops()
    test_pipeline_debug_mode_cifar10_rename_zip(plot=False)
    teardown_function()
