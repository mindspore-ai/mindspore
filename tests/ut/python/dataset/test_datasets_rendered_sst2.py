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
"""
Test RenderedSST2 dataset operators
"""
from multiprocessing import cpu_count
import numpy as np
import os
import platform
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger

IMAGE_DATA_DIR = "../data/dataset/testRenderedSST2Data"
WRONG_DIR = "../data/dataset/notExist"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


def test_rendered_sst2_basic():
    """
    Feature: RenderedSST2Dataset
    Description: Basic test of RenderedSST2Dataset
    Expectation: The data is processed successfully
    """
    logger.info("Test RenderedSST2Dataset Op")
    # case 1: test read all data
    all_data_1 = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shuffle=False)
    all_data_2 = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shuffle=False)

    num_iter = 0
    for item1, item2 in zip(all_data_1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            all_data_2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1["label"], item2["label"])
        num_iter += 1
    assert num_iter == 12

    # case 2: test decode
    all_data_1 = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, decode=True, shuffle=False)
    all_data_2 = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, decode=True, shuffle=False)

    num_iter = 0
    for item1, item2 in zip(all_data_1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            all_data_2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1["label"], item2["label"])
        num_iter += 1
    assert num_iter == 12

    # case 3: test num_samples
    all_data = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, num_samples=4)
    num_iter = 0
    for _ in all_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 4

    # case 4: test repeat
    all_data = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, num_samples=4)
    all_data = all_data.repeat(2)
    num_iter = 0
    for _ in all_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 8

    # case 5: test get_dataset_size, resize and batch
    all_data = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, num_samples=8)
    all_data = all_data.map(operations=[vision.Decode(), vision.Resize((256, 256))], input_columns=["image"],
                            num_parallel_workers=1)

    assert all_data.get_dataset_size() == 8
    assert all_data.get_batch_size() == 1
    # drop_remainder is default to be False
    all_data = all_data.batch(batch_size=3)
    assert all_data.get_batch_size() == 3
    assert all_data.get_dataset_size() == 3

    num_iter = 0
    for _ in all_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 3


def test_rendered_sst2_decode():
    """
    Feature: RenderedSST2Dataset
    Description: Validate RenderedSST2Dataset with decode
    Expectation: The data is processed successfully
    """
    logger.info("Validate RenderedSST2Dataset with decode")
    # define parameters
    repeat_count = 1

    data1 = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, decode=True)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 12


def test_rendered_sst2_sequential_sampler():
    """
    Feature: RenderedSST2Dataset
    Description: Test RenderedSST2Dataset with SequentialSampler
    Expectation: The data is processed successfully
    """
    logger.info("Test RenderedSST2Dataset Op with SequentialSampler")
    num_samples = 4
    sampler = ds.SequentialSampler(num_samples=num_samples)
    all_data_1 = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, sampler=sampler)
    all_data_2 = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shuffle=False, num_samples=num_samples)
    label_list_1, label_list_2 = [], []
    num_iter = 0
    for item1, item2 in zip(all_data_1.create_dict_iterator(num_epochs=1),
                            all_data_2.create_dict_iterator(num_epochs=1)):
        label_list_1.append(item1["label"].asnumpy())
        label_list_2.append(item2["label"].asnumpy())
        num_iter += 1
    np.testing.assert_array_equal(label_list_1, label_list_2)
    assert num_iter == num_samples


def test_rendered_sst2_random_sampler():
    """
    Feature: RenderedSST2Dataset
    Description: Test RenderedSST2Dataset with RandomSampler
    Expectation: The data is processed successfully
    """
    logger.info("Test RenderedSST2Dataset Op with RandomSampler")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    sampler = ds.RandomSampler()
    data1 = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, sampler=sampler)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 12


def test_rendered_sst2_exception():
    """
    Feature: RenderedSST2Dataset
    Description: Test error cases for RenderedSST2Dataset
    Expectation: Throw correct error and message
    """
    logger.info("Test error cases for RenderedSST2Dataset")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shuffle=False, sampler=ds.SequentialSampler(1))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, sampler=ds.SequentialSampler(1), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, num_shards=5, shard_id=-1)

    with pytest.raises(ValueError, match=error_msg_5):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, num_shards=5, shard_id=5)

    with pytest.raises(ValueError, match=error_msg_5):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shuffle=False, num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shuffle=False, num_parallel_workers=-2)

    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, num_shards=2, shard_id="0")

    error_msg_8 = "does not exist or is not a directory or permission denied!"
    with pytest.raises(ValueError, match=error_msg_8):
        all_data = ds.RenderedSST2Dataset(WRONG_DIR)
        for _ in all_data.create_dict_iterator(num_epochs=1):
            pass


def test_rendered_sst2_dataset_operation_01():
    """
    Feature: RenderedSST2Dataset operation
    Description: Testing the normal functionality of the RenderedSST2Dataset operator
    Expectation: The Output is equal to the expected output
    """
    # test dataset_dir
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR)
    num_iter = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
        assert data.get('image').dtype == 'uint8'
        assert data.get('label').dtype == 'uint32'
        num_iter += 1
    assert num_iter == 12

    # test usage is train
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage='train')
    num_iter = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
        assert data.get('image').dtype == 'uint8'
        assert data.get('label').dtype == 'uint32'
        num_iter += 1
    assert num_iter == 4

    # test usage is test
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage='test')
    num_iter = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
        num_iter += 1
    assert num_iter == 4

    # test usage is val
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage='val')
    num_iter = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
        num_iter += 1
    assert num_iter == 4

    # test usage is None
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage='all')
    num_iter = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
        num_iter += 1
    assert num_iter == 12

    # test usage is None
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage=None)
    num_iter = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
        num_iter += 1
    assert num_iter == 12

    # test num_samples is 0
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage='test', num_samples=0)
    num_iter = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
        num_iter += 1
    assert num_iter == 4

    # test num_samples is 1
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage='train', num_samples=1)
    num_iter = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
        num_iter += 1
    assert num_iter == 1

    # test num_samples is 4
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage='val', num_samples=4)
    num_iter = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
        num_iter += 1
    assert num_iter == 4

    # test num_samples is 4
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage='val', num_samples=4)
    num_iter = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
        num_iter += 1
    assert num_iter == 4


def test_rendered_sst2_dataset_operation_02():
    """
    Feature: RenderedSST2Dataset operation
    Description: Testing the normal functionality of the RenderedSST2Dataset operator
    Expectation: The Output is equal to the expected output
    """
    # test num_samples is 4
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage='val', num_samples=4)
    num_iter = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
        num_iter += 1
    assert num_iter == 4

    # test num_parallel_workers 1
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="train", num_parallel_workers=1)
    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        i += 1
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
    assert i == 4

    # test num_parallel_workers 8
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", num_parallel_workers=8)
    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        i += 1
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
    assert i == 12

    # test num_parallel_workers 8
    if platform.system() == "Linux":
        dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="train", num_samples=4, num_parallel_workers=40)
        i = 0
        for data in dataset.create_dict_iterator(output_numpy=True):
            i += 1
            assert "image" in str(data.keys())
            assert "label" in str(data.keys())
        assert i == 4

    # test num_parallel_workers cpucount
    num_pw = cpu_count()
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="train", num_samples=4, num_parallel_workers=num_pw)
    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        i += 1
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
    assert i == 4

    # test shuffle_true
    dataset_1 = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shuffle=True)
    dataset_2 = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", shuffle=True)
    num_iter = 0
    for item1, item2 in zip(dataset_1.create_dict_iterator(output_numpy=True),
                            dataset_2.create_dict_iterator(output_numpy=True)):
        if not np.array_equal(item1.get("image"), item2.get("image")) and \
                (not np.array_equal(item1.get("label"), item2.get("label"))):
            assert True
        num_iter += 1
    assert num_iter == 12

    # test shuffle_true
    dataset_1 = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shuffle=False)
    dataset_2 = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", shuffle=False)
    num_iter = 0
    for item1, item2 in zip(dataset_1.create_dict_iterator(output_numpy=True),
                            dataset_2.create_dict_iterator(output_numpy=True)):
        np.testing.assert_array_equal(item1.get("image"), item2.get("image"))
        np.testing.assert_array_equal(item1.get("label"), item2.get("label"))
        num_iter += 1
    assert num_iter == 12

    # test shuffle_true
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="train", shuffle=None)
    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        i += 1
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
    assert i == 4

    # test shuffle_true
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="train", num_samples=1, decode=True)
    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        i += 1
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
    assert i == 1

    # test shuffle_true
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="train", num_samples=1, decode=False)
    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        i += 1
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
    assert i == 1


def test_rendered_sst2_dataset_operation_03():
    """
    Feature: RenderedSST2Dataset operation
    Description: Testing the normal functionality of the RenderedSST2Dataset operator
    Expectation: The Output is equal to the expected output
    """
    # test RenderedSST2Dataset with RandomSampler
    replacement = True
    num_samples = 4
    sampler = ds.RandomSampler(replacement=replacement, num_samples=num_samples)
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", sampler=sampler)
    num_iter = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
        num_iter += 1
    assert num_iter == 4

    # test RenderedSST2Dataset with use sampler op
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", shuffle=False)
    dataset1, _ = dataset.split([0.8, 0.2], False)
    distributed_sampler = ds.DistributedSampler(2, 0)
    dataset1.use_sampler(distributed_sampler)
    i = 0
    for _ in dataset1.create_dict_iterator(output_numpy=True):
        i += 1
    # 12 * 0.8 / 2
    assert i == 5

    # mappable dataset basic_ops verification (use_sampler/split)
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", shuffle=False)
    dataset = dataset.take(5)
    dataset1, dataset2 = dataset.split([4, 1], randomize=False)
    i = 0
    for _ in dataset1.create_dict_iterator(output_numpy=True):
        i += 1
    assert i == 4
    i = 0
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        i += 1
    assert i == 1

    # test num_shards 2 shard_id 0
    num_shards = 2
    shard_id = 0
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", num_shards=num_shards, shard_id=shard_id)
    num_iter = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
        num_iter += 1
    assert num_iter == 6

    # test num_shards 1 shard_id 10
    num_shards = 1
    shard_id = 0
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", num_shards=num_shards, shard_id=shard_id)
    num_iter = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
        num_iter += 1
    assert num_iter == 12

    # test num_shards 12 shard_id 0
    num_shards = 12
    shard_id = 0
    dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", num_shards=num_shards, shard_id=shard_id)
    num_iter = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert "image" in str(data.keys())
        assert "label" in str(data.keys())
        num_iter += 1
    assert num_iter == 1


def test_rendered_sst2_dataset_exception_01():
    """
    Feature: RenderedSST2Dataset operation
    Description: Testing the RenderedSST2Dataset Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # test no params
    with pytest.raises(TypeError, match="missing a required argument: 'dataset_dir'"):
        _ = ds.RenderedSST2Dataset()

    # test more params
    num_samples = 2
    num_parallel_workers = cpu_count()
    shuffle = True
    num_shards = 3
    shard_id = 2
    more_para = None
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'more_para'"):
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", num_samples=num_samples,
                                   num_parallel_workers=num_parallel_workers,
                                   shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, more_para=more_para)

    # test dataset_dir_type_error
    with pytest.raises(TypeError) as e:
        _ = ds.RenderedSST2Dataset([IMAGE_DATA_DIR])
    assert "is not of type [<class 'str'>]" in str(e.value)

    with pytest.raises(TypeError) as e:
        _ = ds.RenderedSST2Dataset(1)
    assert "is not of type [<class 'str'>]" in str(e.value)

    # test dataset_dir null
    with pytest.raises(ValueError, match="does not exist or is not a directory or permission denied!"):
        _ = ds.RenderedSST2Dataset("")

    # test dataset_dir not a dir
    with pytest.raises(ValueError, match="does not exist or is not a directory or permission denied!"):
        _ = ds.RenderedSST2Dataset("iserrordir")

    # test dataset_dir error dir
    wrong_dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "RenderedSST2Data")
    with pytest.raises(ValueError, match="does not exist or is not a directory or permission denied!"):
        _ = ds.RenderedSST2Dataset(wrong_dataset_dir, usage="all")

    # test usage type error
    with pytest.raises(TypeError) as e:
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage=["all"])
    assert "is not of type [<class 'str'>], but got <class 'list'>" in str(e.value)

    with pytest.raises(TypeError) as e:
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage=1)
    assert "is not of type [<class 'str'>], but got <class 'int'>" in str(e.value)

    # test usage null
    with pytest.raises(ValueError) as e:
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage='')
    assert "Input  is not within the valid set of ['val', 'all', 'train', 'test']." in str(e.value)

    # test usage type error
    with pytest.raises(ValueError) as e:
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage='wrongvalue')
    assert "Input  is not within the valid set of ['val', 'all', 'train', 'test']." in str(e.value)

    # test numsamples type error
    with pytest.raises(TypeError) as e:
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, num_samples=[100])
    assert "Argument num_samples with value [100] is not of type [<class 'int'>]" in str(e.value)

    with pytest.raises(TypeError) as e:
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, num_samples="100")
    assert """Argument num_samples with value 100 is not of type [<class 'int'>]""" in str(e.value)

    # test numsamples null
    with pytest.raises(TypeError) as e:
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, num_samples="")
    assert """Argument num_samples with value "" is not of type [<class 'int'>]""" in str(e.value)

    # test numsamples null
    with pytest.raises(ValueError) as e:
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, num_samples=-1)
    assert 'num_samples exceeds the boundary between 0 and 9223372036854775807' in str(e.value)

    # test num_parallel_workers type error
    with pytest.raises(TypeError) as e:
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", num_parallel_workers="1")
    assert "is not of type [<class 'int'>], but got <class 'str'>." in str(e.value)

    with pytest.raises(TypeError) as e:
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", num_parallel_workers=[1])
    assert "is not of type [<class 'int'>], but got <class 'list'>." in str(e.value)

    # test num_parallel_workers null
    with pytest.raises(TypeError) as e:
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", num_parallel_workers="")
    assert "is not of type [<class 'int'>], but got <class 'str'>." in str(e.value)

    # test num_parallel_workers ValueError
    max_cpu = cpu_count()
    with pytest.raises(ValueError, match=f"num_parallel_workers exceeds the boundary between 1 and {max_cpu}!"):
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", num_parallel_workers=-1)
    with pytest.raises(ValueError, match=f"num_parallel_workers exceeds the boundary between 1 and {max_cpu}!"):
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", num_parallel_workers=0)
    with pytest.raises(ValueError, match=f"num_parallel_workers exceeds the boundary between 1 and {max_cpu}!"):
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", num_parallel_workers=int(max_cpu + 1))


def test_rendered_sst2_dataset_exception_02():
    """
    Feature: RenderedSST2Dataset operation
    Description: Testing the RenderedSST2Dataset Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # test shuffle_type_error
    with pytest.raises(TypeError) as e:
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shuffle="1")
    assert "is not of type [<class 'bool'>], but got <class 'str'>." in str(e.value)

    with pytest.raises(TypeError) as e:
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shuffle=1)
    assert "is not of type [<class 'bool'>], but got <class 'int'>." in str(e.value)

    with pytest.raises(TypeError) as e:
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shuffle=[1])
    assert "is not of type [<class 'bool'>], but got <class 'list'>." in str(e.value)

    # test shuffle_type_null
    with pytest.raises(TypeError) as e:
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shuffle="")
    assert "is not of type [<class 'bool'>], but got <class 'str'>." in str(e.value)

    # test ampler and shuffle are specified at the same time
    with pytest.raises(RuntimeError, match="sampler and shuffle cannot be specified at the same time"):
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", shuffle=False, sampler=ds.SequentialSampler(1))

    # test sampler and sharding are specified at the same time
    with pytest.raises(RuntimeError, match="sampler and sharding cannot be specified at the same time"):
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", sampler=ds.SequentialSampler(1), num_shards=2,
                                   shard_id=0)

    # test sampler wrong
    with pytest.raises(TypeError) as e:
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", sampler='sampler')
    assert "Unsupported sampler object of type (<class 'str'>)" in str(e.value)

    # test num_shards 0 shard_id 0
    num_shards = 0
    shard_id = 0
    with pytest.raises(ValueError) as e:
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", num_shards=num_shards, shard_id=shard_id)
    assert "Input num_shards is not within the required interval of [1, 2147483647]." in str(e.value)

    # test num_shards 1 shard_id 1
    num_shards = 1
    shard_id = 1
    with pytest.raises(ValueError) as e:
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", num_shards=num_shards, shard_id=shard_id)
    assert "Input shard_id is not within the required interval of [0, 0]." in str(e.value)

    # test num_shards 0
    num_shards = 1
    with pytest.raises(RuntimeError) as e:
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", num_shards=num_shards)
    assert "num_shards is specified and currently requires shard_id as well" in str(e.value)

    # test  shard_id 0
    shard_id = 0
    with pytest.raises(RuntimeError) as e:
        _ = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", shard_id=shard_id)
    assert "shard_id is specified but num_shards is not." in str(e.value)

    # test cache
    test_cache = ds.DatasetCache(session_id=41748689, size=0, spilling=False)
    if platform.system() == "Windows":
        with pytest.raises(RuntimeError) as e:
            dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", num_samples=4, cache=test_cache)
            num_iter = 0
            for _ in dataset.create_dict_iterator(num_epochs=1):
                num_iter += 1
            assert "Exception thrown from dataset pipeline. Refer to 'Dataset Pipeline Error Message" in str(e.value)
    else:
        with pytest.raises(RuntimeError) as e:
            dataset = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, usage="all", num_samples=4, cache=test_cache)
            num_iter = 0
            for _ in dataset.create_dict_iterator(num_epochs=1):
                num_iter += 1
            assert "Make sure the server is running" in str(e.value)


if __name__ == '__main__':
    test_rendered_sst2_basic()
    test_rendered_sst2_decode()
    test_rendered_sst2_sequential_sampler()
    test_rendered_sst2_random_sampler()
    test_rendered_sst2_exception()
    test_rendered_sst2_dataset_operation_01()
    test_rendered_sst2_dataset_operation_02()
    test_rendered_sst2_dataset_operation_03()
    test_rendered_sst2_dataset_exception_01()
    test_rendered_sst2_dataset_exception_02()
