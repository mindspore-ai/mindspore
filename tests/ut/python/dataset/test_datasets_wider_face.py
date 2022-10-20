# Copyright 2021 Huawei Technologies Co., Ltd
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
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.log as logger

DATA_DIR = "../data/dataset/testWIDERFace/"


def test_wider_face_basic():
    """
    Feature: WIDERFace dataset
    Description: Read all files
    Expectation: Throw number of data in all files
    """
    logger.info("Test WIDERFaceDataset Op")

    # case 1: test loading default usage dataset
    data1 = ds.WIDERFaceDataset(DATA_DIR)
    num_iter1 = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter1 += 1
    assert num_iter1 == 4

    # case 2: test num_samples
    data2 = ds.WIDERFaceDataset(DATA_DIR, num_samples=1)
    num_iter2 = 0
    for _ in data2.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter2 += 1
    assert num_iter2 == 1

    # case 3: test repeat
    data3 = ds.WIDERFaceDataset(DATA_DIR, num_samples=2)
    data3 = data3.repeat(5)
    num_iter3 = 0
    for _ in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter3 += 1
    assert num_iter3 == 10


def test_wider_face_noshuffle():
    """
    Feature: WIDERFace dataset
    Description: Test noshuffle
    Expectation: Throw number of data in all files
    """
    logger.info("Test Case noShuffle")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    # Note: "all" reads both "train" dataset (2 samples) and "valid" dataset (2 samples)
    data1 = ds.WIDERFaceDataset(DATA_DIR, shuffle=False)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1

    assert num_iter == 4


def test_wider_face_usage():
    """
    Feature: WIDERFace dataset
    Description: Test Usage
    Expectation: Throw number of data in all files
    """
    logger.info("Test WIDERFaceDataset usage flag")

    def test_config(usage, wider_face_path=DATA_DIR):
        try:
            data = ds.WIDERFaceDataset(wider_face_path, usage=usage)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    # test the usage of WIDERFacce
    assert test_config("test") == 3
    assert test_config("train") == 2
    assert test_config("valid") == 2
    assert test_config("all") == 4
    assert "usage is not within the valid set of ['train', 'test', 'valid', 'all']" in test_config(
        "invalid")

    # change to the folder that contains all WIDERFacce files
    all_wider_face = None
    if all_wider_face is not None:
        assert test_config("test", all_wider_face) == 16097
        assert test_config("valid", all_wider_face) == 3226
        assert test_config("train", all_wider_face) == 12880
        assert test_config("all", all_wider_face) == 16106
        assert ds.WIDERFaceDataset(all_wider_face, usage="test").get_dataset_size() == 16097
        assert ds.WIDERFaceDataset(all_wider_face, usage="valid").get_dataset_size() == 3226
        assert ds.WIDERFaceDataset(all_wider_face, usage="train").get_dataset_size() == 12880
        assert ds.WIDERFaceDataset(all_wider_face, usage="all").get_dataset_size() == 16106


def test_wider_face_sequential_sampler():
    """
    Feature: WIDERFace dataset
    Description: Test SequentialSampler
    Expectation: Get correct number of data
    """
    num_samples = 1
    sampler = ds.SequentialSampler(num_samples=num_samples)
    data1 = ds.WIDERFaceDataset(DATA_DIR, 'test', sampler=sampler)
    data2 = ds.WIDERFaceDataset(DATA_DIR, 'test', shuffle=False, num_samples=num_samples)
    matches_list1, matches_list2 = [], []
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1), data2.create_dict_iterator(num_epochs=1)):
        matches_list1.append(item1["image"].asnumpy())
        matches_list2.append(item2["image"].asnumpy())
        num_iter += 1
    np.testing.assert_array_equal(matches_list1, matches_list2)
    assert num_iter == num_samples


def test_wider_face_pipeline():
    """
    Feature: Pipeline test
    Description: Read a sample
    Expectation: The amount of each function are equal
    """
    dataset = ds.WIDERFaceDataset(DATA_DIR, "valid", num_samples=1, decode=True)
    resize_op = vision.Resize((100, 100))
    dataset = dataset.map(input_columns=["image"], operations=resize_op)
    num_iter = 0
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    assert num_iter == 1


def test_wider_face_exception():
    """
    Feature: WIDERFace dataset
    Description: Throw error messages when certain errors occur
    Expectation: Error message
    """
    logger.info("Test error cases for WIDERFaceDataset")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.WIDERFaceDataset(DATA_DIR, shuffle=False, sampler=ds.PKSampler(3))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.WIDERFaceDataset(DATA_DIR, sampler=ds.PKSampler(
            3), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.WIDERFaceDataset(DATA_DIR, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.WIDERFaceDataset(DATA_DIR, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.WIDERFaceDataset(DATA_DIR, num_shards=5, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.WIDERFaceDataset(DATA_DIR, num_shards=5, shard_id=5)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.WIDERFaceDataset(DATA_DIR, num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.WIDERFaceDataset(DATA_DIR, shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.WIDERFaceDataset(DATA_DIR, shuffle=False, num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.WIDERFaceDataset(DATA_DIR, shuffle=False, num_parallel_workers=-2)

    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.WIDERFaceDataset(DATA_DIR, num_shards=2, shard_id="0")

    def exception_func(item):
        raise Exception("Error occur!")

    # usage = test
    try:
        data = ds.WIDERFaceDataset(DATA_DIR, usage="test", shuffle=False)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    # usage = all
    try:
        data = ds.WIDERFaceDataset(DATA_DIR, usage="all", shuffle=False)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.WIDERFaceDataset(DATA_DIR, shuffle=False)
        data = data.map(operations=exception_func, input_columns=["bbox"], num_parallel_workers=1)
        for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.WIDERFaceDataset(DATA_DIR, shuffle=False)
        data = data.map(operations=exception_func, input_columns=["blur"], num_parallel_workers=1)
        for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.WIDERFaceDataset(DATA_DIR, shuffle=False)
        data = data.map(operations=exception_func, input_columns=["expression"], num_parallel_workers=1)
        for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.WIDERFaceDataset(DATA_DIR, shuffle=False)
        data = data.map(operations=exception_func, input_columns=["illumination"], num_parallel_workers=1)
        for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.WIDERFaceDataset(DATA_DIR, shuffle=False)
        data = data.map(operations=exception_func, input_columns=["occlusion"], num_parallel_workers=1)
        for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.WIDERFaceDataset(DATA_DIR, shuffle=False)
        data = data.map(operations=exception_func, input_columns=["pose"], num_parallel_workers=1)
        for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.WIDERFaceDataset(DATA_DIR, shuffle=False)
        data = data.map(operations=exception_func, input_columns=["invalid"], num_parallel_workers=1)
        for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)


if __name__ == '__main__':
    test_wider_face_basic()
    test_wider_face_sequential_sampler()
    test_wider_face_noshuffle()
    test_wider_face_usage()
    test_wider_face_pipeline()
    test_wider_face_exception()
