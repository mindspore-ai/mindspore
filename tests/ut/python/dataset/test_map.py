# Copyright 2023 Huawei Technologies Co., Ltd
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
Test Map op in Dataset
"""
import os
import random
import subprocess
import time
import numpy as np
import psutil
import pytest
import re

import mindspore.dataset as ds
import mindspore.dataset.text as text
from mindspore.dataset.transforms import transforms
import mindspore.dataset.vision as vision
from util import config_get_set_seed, config_get_set_num_parallel_workers, config_get_set_enable_shared_mem

DATA_DIR = "../data/dataset/testPK/data"


def test_map_c_transform_exception():
    """
    Feature: Test Cpp error op def
    Description: Op defined like vision.HWC2CHW
    Expectation: Success
    """
    data_set = ds.ImageFolderDataset(DATA_DIR, num_parallel_workers=1, shuffle=True)

    train_image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    random_crop_decode_resize_op = vision.RandomCropDecodeResize(train_image_size,
                                                                 scale=(0.08, 1.0),
                                                                 ratio=(0.75, 1.333))
    random_horizontal_flip_op = vision.RandomHorizontalFlip(prob=0.5)
    normalize_op = vision.Normalize(mean=mean, std=std)
    hwc2chw_op = vision.HWC2CHW  # exception

    data_set = data_set.map(operations=random_crop_decode_resize_op, input_columns="image", num_parallel_workers=1)
    data_set = data_set.map(operations=random_horizontal_flip_op, input_columns="image", num_parallel_workers=1)
    data_set = data_set.map(operations=normalize_op, input_columns="image", num_parallel_workers=1)
    with pytest.raises(ValueError) as info:
        data_set = data_set.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=1)
    assert "Parameter operations's element of method map should be a " in str(info.value)

    # compose exception
    with pytest.raises(ValueError) as info:
        transforms.Compose([
            vision.RandomCropDecodeResize(train_image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            vision.RandomHorizontalFlip,
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()])
    assert " should be a " in str(info.value)

    # randomapply exception
    with pytest.raises(ValueError) as info:
        transforms.RandomApply([
            vision.RandomCropDecodeResize,
            vision.RandomHorizontalFlip(prob=0.5),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()])
    assert " should be a " in str(info.value)

    # randomchoice exception
    with pytest.raises(ValueError) as info:
        transforms.RandomChoice([
            vision.RandomCropDecodeResize(train_image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.Normalize,
            vision.HWC2CHW()])
    assert " should be a " in str(info.value)


def test_map_py_transform_exception():
    """
    Feature: Test Python error op def
    Description: Op defined like vision.RandomHorizontalFlip
    Expectation: Success
    """
    data_set = ds.ImageFolderDataset(DATA_DIR, num_parallel_workers=1, shuffle=True)

    # define map operations
    decode_op = vision.Decode(to_pil=True)
    random_horizontal_flip_op = vision.RandomHorizontalFlip  # exception
    to_tensor_op = vision.ToTensor()
    trans = [decode_op, random_horizontal_flip_op, to_tensor_op]

    with pytest.raises(ValueError) as info:
        data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=1)
    assert "Parameter operations's element of method map should be a " in str(info.value)

    # compose exception
    with pytest.raises(ValueError) as info:
        transforms.Compose([
            vision.Decode,
            vision.RandomHorizontalFlip(),
            vision.ToTensor()])
    assert " should be a " in str(info.value)

    # randomapply exception
    with pytest.raises(ValueError) as info:
        transforms.RandomApply([
            vision.Decode(to_pil=True),
            vision.RandomHorizontalFlip,
            vision.ToTensor()])
    assert " should be a " in str(info.value)

    # randomchoice exception
    with pytest.raises(ValueError) as info:
        transforms.RandomChoice([
            vision.Decode(to_pil=True),
            vision.RandomHorizontalFlip(),
            vision.ToTensor])
    assert " should be a " in str(info.value)


def test_map_text_and_data_transforms():
    """
    Feature: Map op
    Description: Test Map op with both Text Transforms and Data Transforms
    Expectation: Dataset pipeline runs successfully and results are verified
    """
    data = ds.TextFileDataset("../data/dataset/testVocab/words.txt", shuffle=False)

    vocab = text.Vocab.from_dataset(data, "text", freq_range=None, top_k=None,
                                    special_tokens=["<pad>", "<unk>"],
                                    special_first=True)

    padend_op = transforms.PadEnd([100], pad_value=vocab.tokens_to_ids('<pad>'))
    lookup_op = text.Lookup(vocab, "<unk>")

    # Use both Text Lookup op and Data Transforms PadEnd op in operations list for Map
    data = data.map(operations=[lookup_op, padend_op], input_columns=["text"])
    res = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        res.append(d["text"].item())
    assert res == [4, 5, 3, 6, 7, 2], res


def test_map_operations1():
    """
    Feature: Map op
    Description: Test Map op with operations in multiple formats
    Expectation: Dataset pipeline runs successfully and results are verified
    """

    class RandomHorizontal(vision.RandomHorizontalFlip):
        def __init__(self, p):
            self.p = p
            super().__init__(p)

    data1 = ds.ImageFolderDataset(DATA_DIR, num_samples=5)
    # Use 2 different formats to list ops for map operations
    data1 = data1.map(operations=[vision.Decode(to_pil=True),
                                  vision.RandomCrop(512),
                                  RandomHorizontal(0.5)], input_columns=["image"])

    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1):  # each data is a dictionary
        num_iter += 1
    assert num_iter == 5


def test_c_map_randomness_repeatability(set_seed_to=1111, set_num_parallel_workers_to=3, num_repeat=5):
    """
    Feature: Map op
    Description: Test repeatability of Map op with C implemented random ops with num_parallel_workers > 1
    Expectation: The dataset would be the same each iteration
    """
    data_dir_tf = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
    schema_dir_tf = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
    original_seed = config_get_set_seed(set_seed_to)
    original_num_parallel_workers = config_get_set_num_parallel_workers(set_num_parallel_workers_to)

    # First dataset
    data1 = ds.TFRecordDataset(data_dir_tf, schema_dir_tf, columns_list=["image"], shuffle=False)
    transforms_list1 = [vision.Decode(),
                        vision.RandomResizedCrop((256, 512), (2, 2), (1, 3)),
                        vision.RandomColorAdjust(
                            brightness=(0.5, 0.5), contrast=(0.5, 0.5), saturation=(0.5, 0.5), hue=(0, 0))]
    data1 = data1.map(operations=transforms_list1, input_columns=["image"])

    for _ in range(num_repeat):
        # Next datasets
        data2 = ds.TFRecordDataset(data_dir_tf, schema_dir_tf, columns_list=["image"], shuffle=False)
        transforms_list2 = [vision.Decode(),
                            vision.RandomResizedCrop((256, 512), (2, 2), (1, 3)),
                            vision.RandomColorAdjust(
                                brightness=(0.5, 0.5), contrast=(0.5, 0.5), saturation=(0.5, 0.5), hue=(0, 0))]
        data2 = data2.map(operations=transforms_list2, input_columns=["image"])

        # Expect to have the same image every time
        for img1, img2 in zip(data1.create_tuple_iterator(num_epochs=1, output_numpy=True),
                              data2.create_tuple_iterator(num_epochs=1, output_numpy=True)):
            np.testing.assert_equal(img1, img2)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_c_map_randomness_repeatability_with_shards(set_seed_to=312, set_num_parallel_workers_to=5, num_repeat=5):
    """
    Feature: Map op
    Description: Test repeatability of Map op with C implemented random ops with num_parallel_workers > 1 and sharding
    Expectation: The dataset would be the same each iteration
    """
    image_folder_dir = "../data/dataset/testPK/data"
    num_samples = 55
    num_shards = 2
    shard_id = 0
    shuffle = False
    class_index = dict()
    original_seed = config_get_set_seed(set_seed_to)
    original_num_parallel_workers = config_get_set_num_parallel_workers(set_num_parallel_workers_to)

    # First dataset
    data1 = ds.ImageFolderDataset(image_folder_dir, num_samples=num_samples, num_shards=num_shards,
                                  shard_id=shard_id,
                                  shuffle=shuffle, class_indexing=class_index)
    transforms_list1 = [vision.Decode(),
                        vision.RandomResizedCrop((256, 512), (2, 2), (1, 3)),
                        vision.RandomColorAdjust(
                            brightness=(0.5, 0.5), contrast=(0.5, 0.5), saturation=(0.5, 0.5), hue=(0, 0))]
    data1 = data1.map(operations=transforms_list1, input_columns=["image"])

    for _ in range(num_repeat):
        # Next datasets
        data2 = ds.ImageFolderDataset(image_folder_dir, num_samples=num_samples, num_shards=num_shards,
                                      shard_id=shard_id,
                                      shuffle=shuffle, class_indexing=class_index)
        transforms_list2 = [vision.Decode(),
                            vision.RandomResizedCrop((256, 512), (2, 2), (1, 3)),
                            vision.RandomColorAdjust(
                                brightness=(0.5, 0.5), contrast=(0.5, 0.5), saturation=(0.5, 0.5), hue=(0, 0))]
        data2 = data2.map(operations=transforms_list2, input_columns=["image"])

        # Expect to have the same image every time
        for img1, img2 in zip(data1.create_tuple_iterator(num_epochs=1, output_numpy=True),
                              data2.create_tuple_iterator(num_epochs=1, output_numpy=True)):
            np.testing.assert_equal(img1, img2)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


@pytest.mark.parametrize("num_parallel_workers", (2, 4, 6))
@pytest.mark.parametrize("num_samples", (1, 2, 5, 6))
def test_python_map_mp_repeatability(num_parallel_workers, num_samples, set_seed_to=1605):
    """
    Feature: Map op
    Description: Test repeatability of Map op with Python multiprocessing with Python implemented
    random ops and num_parallel_workers > 1
    Expectation: The dataset would be the same each iteration
    """
    data_dir = "../data/dataset/testImageNetData2/train/"
    original_seed = config_get_set_seed(set_seed_to)
    original_num_parallel_workers = config_get_set_num_parallel_workers(num_parallel_workers)
    # Reduce memory required by disabling the shared memory optimization
    original_enable_shared_mem = config_get_set_enable_shared_mem(False)

    # dataset
    data1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False, num_samples=num_samples)
    transforms_list1 = [vision.Decode(to_pil=True),
                        vision.RandomPerspective(0.4, 1.0),
                        vision.RandomLighting(0.01)]
    data1 = data1.map(transforms_list1, num_parallel_workers=num_parallel_workers, python_multiprocessing=True)

    # Expect to have the same augmentations
    for img1, img2 in zip(data1.create_tuple_iterator(num_epochs=1, output_numpy=True),
                          data1.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_equal(img1, img2)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_enable_shared_mem(original_enable_shared_mem)


def test_python_map_mp_seed_repeatability(set_seed_to=1337, set_num_parallel_workers_to=4, num_repeat=5):
    """
    Feature: Map op
    Description: Test repeatability of Map op with Python multiprocessing with num_parallel_workers > 1
    Expectation: The set of seeds of each process would be the same as expected
    """

    # Generate md int numpy array from [[0, 1], [2, 3]] to [[63, 64], [65, 66]]
    def generator_md():
        for i in range(64):
            yield (np.array([[i, i + 1], [i + 2, i + 3]]),)

    original_seed = config_get_set_seed(set_seed_to)
    original_num_parallel_workers = config_get_set_num_parallel_workers(set_num_parallel_workers_to)
    # Reduce memory required by disabling the shared memory optimization
    original_enable_shared_mem = config_get_set_enable_shared_mem(False)

    expected_result_np_array = {i: [] for i in range(set_seed_to, set_seed_to + set_num_parallel_workers_to)}
    data1 = ds.GeneratorDataset(generator_md, ["data"])
    data1 = data1.map([lambda x: [ds.config.get_seed()] + [random.randrange(1, 1000) for i in range(100)]],
                      num_parallel_workers=set_num_parallel_workers_to, python_multiprocessing=True)
    for item1 in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        seed_used1 = int(list(item1.values())[0][0])
        result_np_array1 = list(item1.values())[0]
        try:
            expected_result_np_array[seed_used1].append(result_np_array1)
        except KeyError:
            raise AssertionError("Not all expected seeds were used")

    for _ in range(num_repeat):
        expected_seed = {i: 0 for i in range(set_seed_to, set_seed_to + set_num_parallel_workers_to)}
        data2 = ds.GeneratorDataset(generator_md, ["data"])
        data2 = data2.map([lambda x: [ds.config.get_seed()] + [random.randrange(1, 1000) for i in range(100)]],
                          num_parallel_workers=set_num_parallel_workers_to, python_multiprocessing=True)
        for item2 in data2.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            seed_used2 = int(list(item2.values())[0][0])
            result_np_array2 = list(item2.values())[0]
            if seed_used2 in expected_seed:
                cur_iter = expected_seed[seed_used2]
                np.testing.assert_array_equal(result_np_array2, expected_result_np_array[seed_used2][cur_iter])
                expected_seed[seed_used2] += 1
            else:
                raise AssertionError("Seed not found")

        if 0 in expected_seed.values():
            raise AssertionError("Not all expected seeds were used")

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_enable_shared_mem(original_enable_shared_mem)


def test_map_just_exchange_columns():
    """
    Feature: Map op
    Description: map with exchange columns pyfunc
    Expectation: success
    """
    # construct the data
    data1 = np.array(np.random.sample(size=(300, 300, 3)) * 255, dtype=np.uint8)
    data2 = np.array(np.random.sample(size=(300, 300, 3)) * 255, dtype=np.uint8)
    data3 = np.array(np.random.sample(size=(300, 300, 3)) * 255, dtype=np.uint8)
    data4 = np.array(np.random.sample(size=(300, 300, 3)) * 255, dtype=np.uint8)

    label = [1, 2, 3, 4]

    # dataset with two columns
    dataset = ds.NumpySlicesDataset(([data1, data2, data3, data4], label), ["data", "label"])

    def exchange_columns(col1, col2):
        return col2, col1

    dataset = dataset.map(operations=exchange_columns, input_columns=["data", "label"],
                          output_columns=["label", "data"])

    for item in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        assert len(item.keys()) == 2
        assert "label" in item.keys()
        assert "data" in item.keys()

    for item in dataset.create_tuple_iterator(output_numpy=True, num_epochs=1):
        assert len(item) == 2
        assert item[0].shape == ()
        assert item[1].shape == (300, 300, 3)

    # dataset with three columns
    dataset2 = ds.NumpySlicesDataset(([data1, data2, data3, data4], [data1, data2, data3, data4], label),
                                     ["data", "data2", "label"])
    dataset2 = dataset2.map(operations=vision.RandomCrop(size=(250, 250)), input_columns="data2")

    def exchange_columns_three(col1, col2, col3):
        return col2, col3, col1

    dataset2 = dataset2.map(operations=exchange_columns_three, input_columns=["data", "data2", "label"],
                            output_columns=["data2", "label", "data"])

    for item in dataset2.create_dict_iterator(output_numpy=True, num_epochs=1):
        assert len(item.keys()) == 3
        assert "label" in item.keys()
        assert "data" in item.keys()
        assert "data2" in item.keys()

    for item in dataset2.create_tuple_iterator(output_numpy=True, num_epochs=1):
        assert len(item) == 3
        assert item[0].shape == (250, 250, 3)
        assert item[1].shape == ()
        assert item[2].shape == (300, 300, 3)


class FakeData:
    def __init__(self):
        self.input_ids = np.ones((128, 128), dtype=np.int32)
        self.input_mask = np.ones((128, 128), dtype=np.int32)

    def __getitem__(self, index):
        return self.input_ids, self.input_mask

    def __len__(self):
        return 791


def test_map_multiprocessing_without_thread():
    """
    Feature: Map op
    Description: map with multiprocessing and don't degenerate into threading
    Expectation: success
    """

    dataset = ds.GeneratorDataset(FakeData(), ["input_ids", "input_mask"])

    def long_running_op(col1, col2):
        data1 = np.ones([50, 3, 655, 655], dtype=np.float64)
        data2 = np.ones([50, 3, 600, 600], dtype=np.float64)
        return data1, data2

    dataset = dataset.map(operations=long_running_op, input_columns=["input_ids", "input_mask"],
                          python_multiprocessing=True, num_parallel_workers=2, max_rowsize=10)
    assert dataset.get_dataset_size() == 791
    assert dataset.output_shapes() == [[50, 3, 655, 655], [50, 3, 600, 600]]
    assert dataset.output_types() == [np.float64, np.float64]
    assert dataset.get_col_names() == ["input_ids", "input_mask"]

    count = 1
    for item in dataset.create_tuple_iterator(output_numpy=True, num_epochs=1):
        assert item[0].dtype == np.float64
        assert item[0].shape == (50, 3, 655, 655)
        assert len(item) == 2
        count += 1
        if count > 5:
            break


def test_map_multiprocessing_with_fixed_handle():
    """
    Feature: Map op
    Description: map with multiprocessing and don't leak pipe handle which is used by queue
    Expectation: success
    """

    dataset = ds.GeneratorDataset(FakeData(), ["input_ids", "input_mask"])

    def long_running_op(col1, col2):
        data1 = np.ones([3, 65, 65], dtype=np.float64)
        data2 = np.ones([3, 60, 60], dtype=np.float64)
        return data1, data2

    dataset = dataset.map(operations=long_running_op, input_columns=["input_ids", "input_mask"],
                          python_multiprocessing=True, num_parallel_workers=4, max_rowsize=10)
    assert dataset.get_dataset_size() == 791

    fds = 0
    for i in range(5):
        count = 0
        for item in dataset.create_tuple_iterator(output_numpy=True, num_epochs=1):
            assert item[0].dtype == np.float64
            assert item[0].shape == (3, 65, 65)
            assert len(item) == 2
            count += 1
        assert count == 791

        # wait for the fds handle to be released automatic
        time.sleep(1)

        i += 1
        if i == 1:
            fds = psutil.Process(os.getpid()).num_fds()
            lsof = subprocess.getoutput("lsof -p " + str(os.getpid()) + " | wc -l")
        elif i > 1:
            assert fds == psutil.Process(os.getpid()).num_fds()
            new_lsof = subprocess.getoutput("lsof -p " + str(os.getpid()) + " | wc -l")
            assert lsof == new_lsof


def test_map_multiprocessing_with_in_out_rowsize_exception():
    """
    Feature: Map op
    Description: map with multiprocessing and max_rowsize with in rowsize & out rowsize exception
    Expectation: success
    """

    dataset = ds.GeneratorDataset(FakeData(), ["input_ids", "input_mask"])

    def long_running_op(col1, col2):
        data1 = np.ones([3, 65, 65], dtype=np.float64)
        data2 = np.ones([3, 60, 60], dtype=np.float64)
        return data1, data2

    with pytest.raises(TypeError) as info:
        dataset = dataset.map(operations=long_running_op, input_columns=["input_ids", "input_mask"],
                              python_multiprocessing=True, num_parallel_workers=2, max_rowsize=(12, 20))
    assert " is not of type " in str(info.value)

    with pytest.raises(TypeError) as info:
        dataset = dataset.map(operations=long_running_op, input_columns=["input_ids", "input_mask"],
                              python_multiprocessing=True, num_parallel_workers=2, max_rowsize="16")
    assert " is not of type " in str(info.value)

    with pytest.raises(TypeError) as info:
        dataset = dataset.map(operations=long_running_op, input_columns=["input_ids", "input_mask"],
                              python_multiprocessing=True, num_parallel_workers=2, max_rowsize=20.5)
    assert " is not of type " in str(info.value)

    with pytest.raises(ValueError) as info:
        dataset = dataset.map(operations=long_running_op, input_columns=["input_ids", "input_mask"],
                              python_multiprocessing=True, num_parallel_workers=2, max_rowsize=-8)
    assert "is not within the required interval of " in str(info.value)

    with pytest.raises(TypeError) as info:
        dataset = dataset.map(operations=long_running_op, input_columns=["input_ids", "input_mask"],
                              python_multiprocessing=True, num_parallel_workers=2, max_rowsize=[12.4, 20])
    assert " is not of type " in str(info.value)

    with pytest.raises(ValueError) as info:
        dataset = dataset.map(operations=long_running_op, input_columns=["input_ids", "input_mask"],
                              python_multiprocessing=True, num_parallel_workers=2, max_rowsize=[-8, 20])
    assert "is not within the required interval of " in str(info.value)

    with pytest.raises(ValueError) as e:
        dataset = dataset.map(operations=long_running_op, input_columns=["input_ids", "input_mask"],
                              python_multiprocessing=True, num_parallel_workers=2, max_rowsize=[-2, 16])
    assert "not within the required interval of [-1, 2147483647]" in str(e.value)

    with pytest.raises(ValueError) as e:
        dataset = dataset.map(operations=long_running_op, input_columns=["input_ids", "input_mask"],
                              python_multiprocessing=True, num_parallel_workers=2, max_rowsize=[16, -5])
    assert "not within the required interval of [-1, 2147483647]" in str(e.value)


def test_map_multiprocessing_with_in_out_rowsize():
    """
    Feature: Map op
    Description: map with multiprocessing and max_rowsize with in rowsize & out rowsize
    Expectation: success
    """

    dataset = ds.GeneratorDataset(FakeData(), ["input_ids", "input_mask"])

    def long_running_op(col1, col2):
        data1 = np.ones([3, 65, 65], dtype=np.float64)
        data2 = np.ones([3, 60, 60], dtype=np.float64)
        return data1, data2

    dataset = dataset.map(operations=long_running_op, input_columns=["input_ids", "input_mask"],
                          python_multiprocessing=True, num_parallel_workers=2, max_rowsize=[3, 6])

    assert dataset.get_dataset_size() == 791

    for _ in range(3):
        count = 0
        for item in dataset.create_tuple_iterator(output_numpy=True, num_epochs=1):
            assert item[0].dtype == np.float64
            assert item[0].shape == (3, 65, 65)
            assert len(item) == 2
            count += 1
        assert count == 791


def test_map_and_generatordataset_with_multiprocessing():
    """
    Feature: Map op
    Description: The output_types and output_shapes methods do not start multiprocessing and multithreading When map or
    GeneratorDataset's parameter multiprocessing is True or num_parallel_workers > 1
    Expectation: The returned result is as expected
    """

    dataset = ds.GeneratorDataset(FakeData(), ["input_ids", "input_mask"], python_multiprocessing=True,
                                  num_parallel_workers=2)

    def long_running_op(col1, col2):
        data1 = np.ones([50, 3, 655, 655], dtype=np.float64)
        data2 = np.ones([50, 3, 600, 600], dtype=np.float64)
        return data1, data2

    dataset = dataset.map(operations=long_running_op, input_columns=["input_ids", "input_mask"],
                          python_multiprocessing=True, num_parallel_workers=2, max_rowsize=10)
    assert dataset.output_shapes() == [[50, 3, 655, 655], [50, 3, 600, 600]]
    assert dataset.output_types() == [np.float64, np.float64]


def map_with_pyfunc_with_multi_ops(mode):
    data_dir = "../data/dataset/testImageNetData2/train"
    data2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)

    def pyfunc2(img_bytes):
        img_decode = vision.Decode()(img_bytes)

        # resize
        img_resize = vision.Resize(size=(64, 32))(img_decode)

        # normalize
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = vision.Normalize(mean=mean_vec, std=std_vec)(img_resize)
        return img_normalize

    # map with PyFunc transform which contains vision.Decode, vision.Resize and vision.Normalize
    data2 = data2.map(pyfunc2, input_columns="image", python_multiprocessing=mode, num_parallel_workers=2)

    for _ in range(5):
        for item in data2.create_tuple_iterator(num_epochs=1, output_numpy=True):
            assert item[0].shape == (64, 32, 3)
            assert item[0].dtype == np.float32

    time.sleep(1)

    # for probably failed
    assert len(transforms.EXECUTORS_LIST) in [0, 1]


class FakeDataWithTransform:
    def __init__(self):
        self.input_ids = np.ones((128, 128, 3), dtype=np.uint8)
        self.input_mask = np.ones((100, 100, 3), dtype=np.int32)

    def __getitem__(self, index):
        img_resize = vision.Resize(size=(64, 32))(self.input_ids)
        return img_resize, self.input_mask

    def __len__(self):
        return 10


def generator_with_multi_transforms(mode):
    # generator with vision.Resize transform
    data2 = ds.GeneratorDataset(source=FakeDataWithTransform(), column_names=["image", "label"], shuffle=False,
                                python_multiprocessing=mode, num_parallel_workers=2)

    def pyfunc2(img):
        # normalize
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = vision.Normalize(mean=mean_vec, std=std_vec)(img)
        return img_normalize

    # map with PyFunc transform which contains vision.Normalize
    data2 = data2.map(pyfunc2, input_columns="image", python_multiprocessing=mode, num_parallel_workers=2)

    for _ in range(5):
        for item in data2.create_tuple_iterator(num_epochs=1, output_numpy=True):
            assert item[0].shape == (64, 32, 3)
            assert item[0].dtype == np.float32

    time.sleep(1)

    # for probably failed
    assert len(transforms.EXECUTORS_LIST) in [0, 1, 2]


def test_generator_or_map_with_pyfunc_use_global_executor():
    """
    Feature: Generator op or Map op with pyfunc contains multi ops which use global executor
    Description: Test generator or map with pyfunc
    Expectation: The result is equal to the expected
    """
    map_with_pyfunc_with_multi_ops(True)
    map_with_pyfunc_with_multi_ops(False)
    generator_with_multi_transforms(True)
    generator_with_multi_transforms(False)


def create_dataset_with_two_workers_randomly_process_identical_images(fix_randomness, transform_type, multiprocessing):
    """Create a dataset that use two workers to randomly process two identical images to test randomness."""
    if fix_randomness:
        ds.config.set_seed(0)

    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    # create a dataset with two identical images
    dataset = ds.NumpySlicesDataset([img, img], column_names=["image"], num_parallel_workers=2)

    def random_add(image):
        """
        Randomly add some values to the image.

        This function is used to test the recovery of randomness of the random
        and numpy libraries in pyfunc.
        """
        image = np.array(image)
        image += random.randint(1, 10)
        image += np.random.randint(1, 10)
        return image

    # if to_pil is False, will use c++ implemented Decode and RandomCrop,
    # otherwise will use python implemented ones
    to_pil = (transform_type == "python")
    transform = [vision.Decode(to_pil=to_pil),
                 vision.RandomCrop((28, 28))]

    # we must add a pyfunc when testing c++ transforms with multiprocessing,
    # because multiprocessing cannot be enabled when only c++ transforms are included
    if transform_type == "python" or multiprocessing:
        transform.append(random_add)

    # create two workers to process these two identical images
    dataset = dataset.map(transform, input_columns=["image"], num_parallel_workers=2,
                          python_multiprocessing=multiprocessing)
    return dataset


@pytest.mark.parametrize("fix_randomness", (False, True))
@pytest.mark.parametrize("transform_type", ("cpp", "python"))
@pytest.mark.parametrize("multiprocessing", (False, True))
def test_randomness_across_workers(fix_randomness, transform_type, multiprocessing):
    """
    Feature: Map
    Description: Test when map concurrent processing is turned on, each worker holds
        its own independent random generator, so the random results are different
    Expectation: Random results are different for each worker
    """
    original_seed = ds.config.get_seed()
    dataset = create_dataset_with_two_workers_randomly_process_identical_images(fix_randomness, transform_type,
                                                                                multiprocessing)
    res = []
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        res.append(data["image"])
    ds.config.set_seed(original_seed)

    # each worker owns independent generator so the results should be different
    assert not np.array_equal(res[0], res[1])


@pytest.mark.parametrize("fix_randomness", (False, True))
@pytest.mark.parametrize("transform_type", ("cpp", "python"))
@pytest.mark.parametrize("multiprocessing", (False, True))
def test_reproducibility_of_random_transforms(fix_randomness, transform_type, multiprocessing):
    """
    Feature: Map
    Description: Test when map concurrent processing is turned on, each worker holds
        its own independent random generator, so the random results are different
    Expectation: Random results are different for each worker
    """
    original_seed = ds.config.get_seed()
    original_prefetch_size = ds.config.get_prefetch_size()
    ds.config.set_prefetch_size(2)
    dataset = create_dataset_with_two_workers_randomly_process_identical_images(fix_randomness, transform_type,
                                                                                multiprocessing)
    # run the pipeline twice, when the random seed is set, the results should be consistent
    # between the two times, otherwise they should be different
    res_first_time = []
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        res_first_time.append(data["image"])
    res_second_time = []
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        res_second_time.append(data["image"])
    ds.config.set_seed(original_seed)
    ds.config.set_prefetch_size(original_prefetch_size)

    # currently, the results of random pyfunc with multi-threads cannot be fixed
    if not multiprocessing and transform_type == "python":
        expect_equal = False
    else:
        expect_equal = fix_randomness
    for (data_first_time, data_second_time) in zip(res_first_time, res_second_time):
        assert np.array_equal(data_first_time, data_second_time) == expect_equal


def test_map_pullmode_exception():
    """
    Feature: Test map in pull mode
    Description: Test map in pull mode and raise exception as expected
    Expectation: Success
    """
    data_set = ds.ImageFolderDataset(DATA_DIR, num_parallel_workers=1, shuffle=True)

    # define map operations
    data_set = data_set.map(lambda x: (x + x), input_columns=["image"], output_columns=["image1", "image2"])

    with pytest.raises(RuntimeError) as e:
        data_set.output_shapes()
    assert "number of columns returned in 'map' operations should match the number of 'output_columns'" in str(e.value)


def test_map_batch_slowly():
    """
    Feature: Test map & batch slowly
    Description: Test map map & batch slowly
    Expectation: Success
    """

    # rm the log file
    log_file = "./test_map_batch_slowly.log"
    if os.path.exists(log_file):
        os.remove(log_file)

    # run the testcase
    cmd = "python dataset/test_map_batch_hung.py >" + log_file + " 2>&1"
    ret = os.system(cmd)
    assert ret == 0

    # grep the log file
    with open(log_file) as read_file:
        line_info = read_file.read()
    assert re.search("map time: 3", line_info)
    assert re.search("batch time: 3", line_info)
    assert re.search("Batch worker subprocess ID .* is stuck", line_info)
    assert re.search("Map worker subprocess ID .* is stuck", line_info)

    ret = os.system("which py-spy")
    if ret == 0:
        assert re.search("Map worker subprocess stack:", line_info)
        assert re.search("Batch worker subprocess stack:", line_info)
    else:
        assert re.search("Please `pip install py-spy` to get the stacks of the stuck process", line_info)

    # rm the log file
    if os.path.exists(log_file):
        os.remove(log_file)


if __name__ == '__main__':
    test_map_c_transform_exception()
    test_map_py_transform_exception()
    test_map_text_and_data_transforms()
    test_map_operations1()
    test_c_map_randomness_repeatability()
    test_c_map_randomness_repeatability_with_shards()
    test_python_map_mp_repeatability(num_parallel_workers=4, num_samples=4)
    test_python_map_mp_seed_repeatability()
    test_map_with_deprecated_parameter()
    test_map_just_exchange_columns()
    test_map_multiprocessing_without_thread()
    test_map_multiprocessing_with_fixed_handle()
    test_map_multiprocessing_with_in_out_rowsize()
    test_map_multiprocessing_with_in_out_rowsize_exception()
    test_map_and_generatordataset_with_multiprocessing()
    test_generator_or_map_with_pyfunc_use_global_executor()
    test_randomness_across_workers(fix_randomness=True, transform_type="cpp", multiprocessing=False)
    test_reproducibility_of_random_transforms(fix_randomness=False, transform_type="python", multiprocessing=True)
    test_map_pullmode_exception()
    test_map_batch_slowly()
