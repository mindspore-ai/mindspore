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
import random
import numpy as np
import pytest

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


# Run this test in separate process since this test updates config settings
@pytest.mark.forked
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


# Run this test in separate process since this test updates config settings
@pytest.mark.forked
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


def test_map_with_deprecated_parameter():
    """
    Feature: Map op
    Description: map with deprecated parameter
    Expectation: ValueError
    """
    data1 = np.array(np.random.sample(size=(300, 300, 3)) * 255, dtype=np.uint8)
    data2 = np.array(np.random.sample(size=(300, 300, 3)) * 255, dtype=np.uint8)
    data3 = np.array(np.random.sample(size=(300, 300, 3)) * 255, dtype=np.uint8)
    data4 = np.array(np.random.sample(size=(300, 300, 3)) * 255, dtype=np.uint8)

    label = [1, 2, 3, 4]

    dataset = ds.NumpySlicesDataset(([data1, data2, data3, data4], label), ["data", "label"])
    with pytest.raises(ValueError) as info:
        dataset = dataset.map(operations=[(lambda x: (x + 1, x / 255))],
                              input_columns=["data"],
                              output_columns=["data2", "data3"],
                              column_order=["data2", "data3"])
    assert "The parameter 'column_order' had been deleted in map operation." in str(info.value)


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
        print(item[0].shape, item[1].shape, item[2].shape)
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
        print("count: {}, type: {}, shape: {}".format(count, item[0].dtype, item[0].shape))
        assert item[0].dtype == np.float64
        assert item[0].shape == (50, 3, 655, 655)
        assert len(item) == 2
        count += 1
        if count > 5:
            break


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
