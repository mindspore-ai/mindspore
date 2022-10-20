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
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision as vision

DATA_DIR_SEMEION = "../data/dataset/testSemeionData"


def load_semeion(path):
    """
    load Semeion data
    """
    fp = os.path.realpath(os.path.join(path, "semeion.data"))
    data = np.loadtxt(fp)

    images = (data[:, :256]).astype('uint8')
    images = images.reshape(-1, 16, 16)
    labels = np.nonzero(data[:, 256:])[1]

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


def test_semeion_content_check():
    """
    Feature: SemeionDataset
    Description: Check content of each sample
    Expectation: Correct content
    """
    data1 = ds.SemeionDataset(DATA_DIR_SEMEION, num_samples=10, shuffle=False)
    images, labels = load_semeion(DATA_DIR_SEMEION)
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label"
    for i, d in enumerate(data1.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(d["image"], images[i])
        np.testing.assert_array_equal(d["label"], labels[i])
        num_iter += 1
    assert num_iter == 10


def test_semeion_basic():
    """
    Feature: SemeionDataset
    Description: Use different data to test the functions of different versions
    Expectation: all samples(10)
                num_samples
                        set   5
                        get   5
                num_parallel_workers
                        set    1(num_samples=6)
                        get   6
                num repeat
                        set    3(num_samples=3)
                        get   9
    """
    # case 0: test loading all samples
    data0 = ds.SemeionDataset(DATA_DIR_SEMEION)
    num_iter0 = 0
    for _ in data0.create_dict_iterator(num_epochs=1):
        num_iter0 += 1
    assert num_iter0 == 10

    # case 1: test num_samples
    data1 = ds.SemeionDataset(DATA_DIR_SEMEION, num_samples=5)
    num_iter1 = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter1 += 1
    assert num_iter1 == 5

    # case 2: test num_parallel_workers
    data2 = ds.SemeionDataset(DATA_DIR_SEMEION, num_samples=6, num_parallel_workers=1)
    num_iter2 = 0
    for _ in data2.create_dict_iterator(num_epochs=1):
        num_iter2 += 1
    assert num_iter2 == 6

    # case 3: test repeat
    data3 = ds.SemeionDataset(DATA_DIR_SEMEION, num_samples=3)
    data3 = data3.repeat(3)
    num_iter3 = 0
    for _ in data3.create_dict_iterator(num_epochs=1):
        num_iter3 += 1
    assert num_iter3 == 9


def test_semeion_sequential_sampler():
    """
    Feature: SemeionDataset
    Description: Test semeion sequential sampler
    Expectation: Correct data
    """
    num_samples = 4
    sampler = ds.SequentialSampler(num_samples=num_samples)
    data1 = ds.SemeionDataset(DATA_DIR_SEMEION, sampler=sampler)
    data2 = ds.SemeionDataset(DATA_DIR_SEMEION, shuffle=False, num_samples=num_samples)
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_equal(item1["label"], item2["label"])
        np.testing.assert_equal(item1["image"], item2["image"])
        num_iter += 1
    assert num_iter == num_samples


def test_semeion_exceptions():
    """
    Feature: SemeionDataset
    Description: Error test
    Expectation: Throw error
    """
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.SemeionDataset(DATA_DIR_SEMEION, shuffle=False, sampler=ds.PKSampler(3))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.SemeionDataset(DATA_DIR_SEMEION, sampler=ds.PKSampler(3), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.SemeionDataset(DATA_DIR_SEMEION, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.SemeionDataset(DATA_DIR_SEMEION, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.SemeionDataset(DATA_DIR_SEMEION, num_shards=2, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.SemeionDataset(DATA_DIR_SEMEION, num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.SemeionDataset(DATA_DIR_SEMEION, shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.SemeionDataset(DATA_DIR_SEMEION, shuffle=False, num_parallel_workers=256)


def test_semeion_visualize(plot=False):
    """
    Feature: SemeionDataset
    Description: Visualize SemeionDataset results
    Expectation: Visualization
    """
    data1 = ds.SemeionDataset(DATA_DIR_SEMEION, num_samples=10, shuffle=False)
    num_iter = 0
    image_list, label_list = [], []
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item["image"]
        label = item["label"]
        image_list.append(image)
        label_list.append("label {}".format(label))
        assert isinstance(image, np.ndarray)
        assert image.shape == (16, 16)
        assert image.dtype == np.uint8
        assert label.dtype == np.uint32
        num_iter += 1
    assert num_iter == 10
    if plot:
        visualize_dataset(image_list, label_list)


def test_semeion_exception_file_path():
    """
    Feature: SemeionDataset
    Description: Error test
    Expectation: Throw error
    """
    def exception_func(item):
        raise Exception("Error occur!")

    try:
        data = ds.SemeionDataset(DATA_DIR_SEMEION)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator():
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.SemeionDataset(DATA_DIR_SEMEION)
        data = data.map(operations=exception_func, input_columns=["label"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator():
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)


def test_semeion_pipeline():
    """
    Feature: SemeionDataset
    Description: Read a sample
    Expectation: The amount of each function are equal
    """
    # Original image
    dataset = ds.SemeionDataset(DATA_DIR_SEMEION, num_samples=1)
    resize_op = vision.Resize((100, 100))
    # Filtered image by Resize
    dataset = dataset.map(operations=resize_op, input_columns=["image"], num_parallel_workers=1)
    i = 0
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        i += 1
    assert i == 1


if __name__ == '__main__':
    test_semeion_content_check()
    test_semeion_basic()
    test_semeion_sequential_sampler()
    test_semeion_exceptions()
    test_semeion_visualize(plot=False)
    test_semeion_exception_file_path()
    test_semeion_pipeline()
