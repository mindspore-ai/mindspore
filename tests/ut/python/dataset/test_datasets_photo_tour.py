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
Test PhotoTour dataset operations
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

import mindspore.dataset as ds
from mindspore import log as logger

DATA_DIR = "../data/dataset/testPhotoTourData"
NAME = 'liberty'
LEN = 100


def load_photo_tour_dataset(path, name):
    """
    Feature: load_photo_tour_dataset.
    Description: Load photo tour.
    Expectation: Get data of photo tour dataset.
    """
    def pil2array(img: Image.Image):
        """
        Convert PIL image type to numpy 2D array
        """
        return np.array(img.getdata(), dtype=np.uint8).reshape((64, 64, 1))

    def find_files(data_dir: str, image_ext_: str):
        """
        Return a list with the file names of the images containing the patches
        """
        files = []
        # find those files with the specified extension
        for file_dir in os.listdir(data_dir):
            if file_dir.endswith(image_ext_):
                files.append(os.path.join(data_dir, file_dir))
        return sorted(files)  # sort files in ascend order to keep relations

    patches = []
    list_files = find_files(os.path.realpath(os.path.join(path, name)), 'bmp')
    idx = 0
    for fpath in list_files:
        img = Image.open(fpath)
        for y in range(0, 1024, 64):
            for x in range(0, 1024, 64):
                patch = img.crop((x, y, x + 64, y + 64))
                patches.append(pil2array(patch))
                idx += 1
                if idx > LEN:
                    break
            if idx > LEN:
                break
    matches_path = os.path.join(os.path.realpath(os.path.join(path, name)), 'm50_100000_100000_0.txt')
    matches = []
    with open(matches_path, 'r') as f:
        for line in f.readlines():
            line_split = line.split()
            matches.append([int(line_split[0]), int(line_split[3]),
                            int(line_split[1] == line_split[4])])
    return patches, matches


def visualize_dataset(images1, images2, matches):
    """
    Feature: visualize_dataset.
    Description: Visualize photo tour dataset.
    Expectation: Plot images.
    """
    num_samples = len(images1)
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images1[i].squeeze(), cmap=plt.cm.gray)
        plt.title(matches[i])
    num_samples = len(images2)
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(images2[i].squeeze(), cmap=plt.cm.gray)
        plt.title(matches[i])
    plt.show()


def test_photo_tour_content_check():
    """
    Feature: test_photo_tour_content_check.
    Description: Validate PhotoTourDataset image readings.
    Expectation: Get correct number of data and correct content.
    """
    logger.info("Test PhotoTourDataset Op with content check")
    data1 = ds.PhotoTourDataset(DATA_DIR, NAME, 'test', num_samples=10, shuffle=False)
    images, matches = load_photo_tour_dataset(DATA_DIR, NAME)
    num_iter = 0
    # in this example, each dictionary has keys "image1" "image2" and "matches"

    for i, data in enumerate(data1.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(data["image1"], images[matches[i][0]])
        np.testing.assert_array_equal(data["image2"], images[matches[i][1]])
        np.testing.assert_array_equal(data["matches"], matches[i][2])
        num_iter += 1
    assert num_iter == 10


def test_photo_tour_basic():
    """
    Feature: test_photo_tour_basic.
    Description: Test basic usage of PhotoTourDataset.
    Expectation: Get correct number of data.
    """
    logger.info("Test PhotoTourDataset Op")

    # case 1: test loading whole dataset
    data1 = ds.PhotoTourDataset(DATA_DIR, NAME, 'test')
    num_iter1 = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter1 += 1
    assert num_iter1 == 16

    # case 2: test num_samples
    data2 = ds.PhotoTourDataset(DATA_DIR, NAME, 'test', num_samples=10)
    num_iter2 = 0
    for _ in data2.create_dict_iterator(num_epochs=1):
        num_iter2 += 1
    assert num_iter2 == 10

    # case 3: test repeat
    data3 = ds.PhotoTourDataset(DATA_DIR, NAME, 'test', num_samples=5)
    data3 = data3.repeat(5)
    num_iter3 = 0
    for _ in data3.create_dict_iterator(num_epochs=1):
        num_iter3 += 1
    assert num_iter3 == 25

    # case 4: test batch with drop_remainder=False
    data4 = ds.PhotoTourDataset(DATA_DIR, NAME, 'test', num_samples=10)
    assert data4.get_dataset_size() == 10
    assert data4.get_batch_size() == 1
    data4 = data4.batch(batch_size=7)  # drop_remainder is default to be False
    assert data4.get_dataset_size() == 2
    assert data4.get_batch_size() == 7
    num_iter4 = 0
    for _ in data4.create_dict_iterator(num_epochs=1):
        num_iter4 += 1
    assert num_iter4 == 2

    # case 5: test batch with drop_remainder=True
    data5 = ds.PhotoTourDataset(DATA_DIR, NAME, 'test', num_samples=10)
    assert data5.get_dataset_size() == 10
    assert data5.get_batch_size() == 1
    data5 = data5.batch(batch_size=7, drop_remainder=True)  # the rest of incomplete batch will be dropped
    assert data5.get_dataset_size() == 1
    assert data5.get_batch_size() == 7
    num_iter5 = 0
    for _ in data5.create_dict_iterator(num_epochs=1):
        num_iter5 += 1
    assert num_iter5 == 1

    # case 6: test get_col_names
    data6 = ds.PhotoTourDataset(DATA_DIR, NAME, 'test', num_samples=10)
    assert data6.get_col_names() == ['image1', 'image2', 'matches']


def test_photo_tour_pk_sampler():
    """
    Feature: test_photo_tour_pk_sampler.
    Description: Test usage of PhotoTourDataset with PKSampler.
    Expectation: Get correct number of data.
    """
    logger.info("Test PhotoTourDataset Op with PKSampler")
    golden = [0, 0, 0, 1, 1, 1]
    sampler = ds.PKSampler(3)
    data = ds.PhotoTourDataset(DATA_DIR, NAME, 'test', sampler=sampler)
    num_iter = 0
    matches_list = []
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        matches_list.append(item["matches"])
        num_iter += 1
    np.testing.assert_array_equal(golden, matches_list)
    assert num_iter == 6


def test_photo_tour_sequential_sampler():
    """
    Feature: test_photo_tour_sequential_sampler.
    Description: Test usage of PhotoTourDataset with SequentialSampler.
    Expectation: Get correct number of data.
    """
    logger.info("Test PhotoTourDataset Op with SequentialSampler")
    num_samples = 5
    sampler = ds.SequentialSampler(num_samples=num_samples)
    data1 = ds.PhotoTourDataset(DATA_DIR, NAME, 'test', sampler=sampler)
    data2 = ds.PhotoTourDataset(DATA_DIR, NAME, 'test', shuffle=False, num_samples=num_samples)
    matches_list1, matches_list2 = [], []
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1), data2.create_dict_iterator(num_epochs=1)):
        matches_list1.append(item1["matches"].asnumpy())
        matches_list2.append(item2["matches"].asnumpy())
        num_iter += 1
    np.testing.assert_array_equal(matches_list1, matches_list2)
    assert num_iter == num_samples


def test_photo_tour_exception():
    """
    Feature: test_photo_tour_exception.
    Description: Test error cases for PhotoTourDataset.
    Expectation: Raise exception.
    """
    logger.info("Test error cases for PhotoTourDataset")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.PhotoTourDataset(DATA_DIR, NAME, 'test', shuffle=False, sampler=ds.PKSampler(3))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.PhotoTourDataset(DATA_DIR, NAME, 'test', sampler=ds.PKSampler(3), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.PhotoTourDataset(DATA_DIR, NAME, 'test', num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.PhotoTourDataset(DATA_DIR, NAME, 'test', shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.PhotoTourDataset(DATA_DIR, NAME, 'test', num_shards=5, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.PhotoTourDataset(DATA_DIR, NAME, 'test', num_shards=5, shard_id=5)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.PhotoTourDataset(DATA_DIR, NAME, 'test', num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.PhotoTourDataset(DATA_DIR, NAME, 'test', shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.PhotoTourDataset(DATA_DIR, NAME, 'test', shuffle=False, num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.PhotoTourDataset(DATA_DIR, NAME, 'test', shuffle=False, num_parallel_workers=-2)

    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.PhotoTourDataset(DATA_DIR, NAME, 'test', num_shards=2, shard_id="0")


def test_photo_tour_visualize(plot=False):
    """
    Feature: test_photo_tour_visualize.
    Description: Visualize PhotoTourDataset results.
    Expectation: Get correct number of data and plot them.
    """
    logger.info("Test PhotoTourDataset visualization")

    data1 = ds.PhotoTourDataset(DATA_DIR, NAME, 'test', num_samples=10, shuffle=False)
    num_iter = 0
    image_list1, image_list2, matches_list = [], [], []
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image1 = item["image1"]
        image2 = item["image2"]
        matches = item["matches"]
        image_list1.append(image1)
        image_list2.append(image2)
        matches_list.append("matches {}".format(matches))
        assert isinstance(image1, np.ndarray)
        assert isinstance(image2, np.ndarray)
        assert image1.shape == (64, 64, 1)
        assert image1.dtype == np.uint8
        assert image2.shape == (64, 64, 1)
        assert image2.dtype == np.uint8
        assert matches.dtype == np.uint32
        num_iter += 1
    assert num_iter == 10
    if plot:
        visualize_dataset(image_list1, image_list2, matches_list)


def test_photo_tour_usage():
    """
    Feature: test_photo_tour_usage.
    Description: Validate PhotoTourDataset image readings.
    Expectation: Get correct number of data.
    """
    logger.info("Test PhotoTourDataset usage flag")

    def test_config(photo_tour_path, name, usage):
        try:
            data = ds.PhotoTourDataset(photo_tour_path, name, usage, shuffle=False)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    assert test_config(DATA_DIR, NAME, "test") == 16
    assert test_config(DATA_DIR, NAME, "train") == LEN
    assert "usage is not within the valid set of ['train', 'test']" in test_config(DATA_DIR, NAME, "invalid")
    assert "Argument usage with value ['list'] is not of type [<class 'str'>]" in test_config(DATA_DIR, NAME, ["list"])


if __name__ == '__main__':
    test_photo_tour_content_check()
    test_photo_tour_basic()
    test_photo_tour_pk_sampler()
    test_photo_tour_sequential_sampler()
    test_photo_tour_exception()
    test_photo_tour_visualize(plot=True)
    test_photo_tour_usage()
