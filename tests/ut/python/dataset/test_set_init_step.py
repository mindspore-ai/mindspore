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

import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision as vision


def init_dataset_from_step_and_verify_result(dataset, init_step, num_epochs, dataset_size, expected_result=None):
    """
    Initialize the given dataset from the specified step and verify the results.
    If expected_result is None, we only check the number of total steps. Or we
    will also verify if the data is the same as expected.
    """
    dataset.set_init_step(init_step=init_step)
    iterator = dataset.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)
    step_count = 0
    for _ in range(num_epochs - init_step // dataset_size):
        for data in iterator:
            if expected_result is not None:
                for column in data.keys():
                    np.testing.assert_array_equal(data[column], expected_result[step_count + init_step][column])
            step_count += 1
    assert step_count == num_epochs * dataset_size - init_step


def get_expected_result(dataset, num_epochs):
    """
    Iterate the whole dataset to save the expected results.
    """
    iterator = dataset.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)
    expected_result = []
    for _ in range(num_epochs):
        for data in iterator:
            expected_result.append(data)
    return expected_result


@pytest.mark.parametrize("fast_recovery_mode", (True, False))
@pytest.mark.parametrize("shuffle", (True, False, ds.Shuffle.GLOBAL, ds.Shuffle.FILES, ds.Shuffle.INFILE))
def test_init_step_with_non_mappable_source(fast_recovery_mode, shuffle):
    """
    Feature: Pipeline resuming
    Description: Initialize TFRecordDataset from intermediate step
    Expectation: Pipeline returns data from the specified step
    """
    original_mode = ds.config.get_fast_recovery()
    ds.config.set_fast_recovery(fast_recovery_mode)
    original_seed = ds.config.get_seed()
    ds.config.set_seed(0)

    dataset_path = "../data/dataset/test_tf_file_3_images2/"
    dataset = ds.TFRecordDataset([dataset_path + "train-0000-of-0001.data",
                                  dataset_path + "train-0000-of-0002.data",
                                  dataset_path + "train-0000-of-0003.data",
                                  dataset_path + "train-0000-of-0004.data"],
                                 dataset_path + "datasetSchema.json",
                                 shuffle=shuffle)
    decode = vision.Decode()
    resize = vision.Resize((32, 32))
    dataset = dataset.map([decode, resize], input_columns=["image"])

    if not fast_recovery_mode:
        # only in non fast recovery mode, we can keep random transforms the same
        random_vertical_flip = vision.RandomVerticalFlip()
        random_color = vision.RandomColor()
        dataset = dataset.map([random_vertical_flip, random_color])

    num_epochs = 2
    dataset_size = dataset.get_dataset_size()
    expected_result = get_expected_result(dataset, num_epochs)
    # test initialize from the beginning, the middle of the first epoch,
    # the end of the first epoch and the middle of the second epoch
    for init_step in [0, dataset_size // 2, dataset_size, dataset_size + dataset_size // 2]:
        init_dataset_from_step_and_verify_result(dataset, init_step, num_epochs, dataset_size, expected_result)

    ds.config.set_fast_recovery(original_mode)
    ds.config.set_seed(original_seed)


@pytest.mark.parametrize("fast_recovery_mode", (True, False))
@pytest.mark.parametrize("shuffle", (True, False))
def test_init_step_with_mappable_source(fast_recovery_mode, shuffle):
    """
    Feature: Pipeline resuming
    Description: Initialize ImageFolderDataset from intermediate step
    Expectation: Pipeline returns data from the specified step
    """
    original_mode = ds.config.get_fast_recovery()
    ds.config.set_fast_recovery(fast_recovery_mode)
    original_seed = ds.config.get_seed()
    ds.config.set_seed(0)

    dataset = ds.ImageFolderDataset("../data/dataset/testPK/data", decode=True, shuffle=shuffle)
    resize = vision.Resize((32, 32))
    dataset = dataset.map([resize], input_columns=["image"])

    if not fast_recovery_mode:
        # only in non fast recovery mode, we can keep random transforms the same
        random_horizontal_flip = vision.RandomHorizontalFlip()
        random_color_adjust = vision.RandomColorAdjust()
        dataset = dataset.map([random_horizontal_flip, random_color_adjust])

    dataset = dataset.batch(4)

    num_epochs = 2
    dataset_size = dataset.get_dataset_size()
    expected_result = get_expected_result(dataset, num_epochs)
    # test initialize from the beginning, the middle of the first epoch,
    # the end of the first epoch and the middle of the second epoch
    for init_step in [0, dataset_size // 2, dataset_size, dataset_size + dataset_size // 2]:
        init_dataset_from_step_and_verify_result(dataset, init_step, num_epochs, dataset_size, expected_result)

    ds.config.set_fast_recovery(original_mode)
    ds.config.set_seed(original_seed)


@pytest.mark.parametrize("fast_recovery_mode", (True, False))
@pytest.mark.parametrize("shuffle", (True, False))
def test_init_step_with_non_mappable_generator(fast_recovery_mode, shuffle):
    """
    Feature: Pipeline resuming
    Description: Initialize non-mappable GeneratorDataset from intermediate step
    Expectation: Pipeline returns data from the specified step
    """
    original_mode = ds.config.get_fast_recovery()
    ds.config.set_fast_recovery(fast_recovery_mode)
    original_seed = ds.config.get_seed()
    ds.config.set_seed(0)

    def gen():
        for i in range(20):
            yield i

    dataset = ds.GeneratorDataset(gen, column_names=["data"])

    if shuffle:
        dataset = dataset.shuffle(2)

    def process(data):
        return data * data

    dataset = dataset.map(process, input_columns=["data"], num_parallel_workers=2, python_multiprocessing=True)
    dataset = dataset.batch(10)

    num_epochs = 2
    dataset_size = dataset.get_dataset_size()
    expected_result = get_expected_result(dataset, num_epochs)
    # test initialize from the beginning, the middle of the first epoch,
    # the end of the first epoch and the middle of the second epoch
    for init_step in [0, dataset_size // 2, dataset_size, dataset_size + dataset_size // 2]:
        init_dataset_from_step_and_verify_result(dataset, init_step, num_epochs, dataset_size, expected_result)

    ds.config.set_fast_recovery(original_mode)
    ds.config.set_seed(original_seed)


def test_init_step_with_non_mappable_generator_with_len():
    """
    Feature: Pipeline resuming
    Description: Initialize non-mappable GeneratorDataset with len method from intermediate step
    Expectation: Pipeline returns data from the specified step
    """
    original_mode = ds.config.get_fast_recovery()
    ds.config.set_fast_recovery(True)
    original_seed = ds.config.get_seed()
    ds.config.set_seed(0)

    class MyDataset:
        def __init__(self, length):
            self.length = length

        def __iter__(self):
            return iter(range(self.length))

        def __len__(self):
            return self.length

    dataset = ds.GeneratorDataset(MyDataset(5), column_names=["data"])

    num_epochs = 2
    dataset_size = dataset.get_dataset_size()
    expected_result = get_expected_result(dataset, num_epochs)
    # test initialize from the beginning, the middle of the first epoch,
    # the end of the first epoch and the middle of the second epoch
    for init_step in [0, dataset_size // 2, dataset_size, dataset_size + dataset_size // 2]:
        init_dataset_from_step_and_verify_result(dataset, init_step, num_epochs, dataset_size, expected_result)

    ds.config.set_fast_recovery(original_mode)
    ds.config.set_seed(original_seed)


@pytest.mark.skip(reason="timeout")
@pytest.mark.parametrize("fast_recovery_mode", (True, False))
@pytest.mark.parametrize("shuffle", (True, False))
def test_init_step_with_mappable_generator(fast_recovery_mode, shuffle):
    """
    Feature: Pipeline resuming
    Description: Initialize mappable GeneratorDataset from intermediate step
    Expectation: Pipeline returns data from the specified step
    """
    original_mode = ds.config.get_fast_recovery()
    ds.config.set_fast_recovery(fast_recovery_mode)
    original_seed = ds.config.get_seed()
    ds.config.set_seed(0)

    class MyDataset:
        def __init__(self, length):
            self.length = length
            self.data = np.random.randint(0, 255, (self.length, 28, 28, 3), dtype=np.uint8)

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return self.length

    dataset = ds.GeneratorDataset(MyDataset(6), column_names=["image"], num_parallel_workers=2)

    if shuffle:
        dataset = dataset.shuffle(6)

    dataset = dataset.batch(3)
    dataset = dataset.repeat(1)

    num_epochs = 2
    dataset_size = dataset.get_dataset_size()
    expected_result = get_expected_result(dataset, num_epochs)
    # test initialize from the beginning, the middle of the first epoch,
    # the end of the first epoch and the middle of the second epoch
    for init_step in [0, dataset_size // 2, dataset_size, dataset_size + dataset_size // 2]:
        init_dataset_from_step_and_verify_result(dataset, init_step, num_epochs, dataset_size, expected_result)

    ds.config.set_fast_recovery(original_mode)
    ds.config.set_seed(original_seed)


@pytest.mark.parametrize("fast_recovery_mode", (True, False))
@pytest.mark.parametrize("shuffle", (True, False))
def test_init_step_with_mind_dataset(fast_recovery_mode, shuffle):
    """
    Feature: Pipeline resuming
    Description: Initialize MindDataset from intermediate step
    Expectation: Pipeline returns data from the specified step
    """
    original_mode = ds.config.get_fast_recovery()
    ds.config.set_fast_recovery(fast_recovery_mode)
    original_seed = ds.config.get_seed()
    ds.config.set_seed(0)

    dataset = ds.MindDataset("../data/mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0",
                             shuffle=shuffle, num_shards=4, shard_id=0)
    decode = vision.Decode()
    dataset = dataset.map([decode], input_columns=["data"], output_columns=["image"])
    resize = vision.Resize((32, 32))
    dataset = dataset.map([resize], input_columns=["image"])

    if not fast_recovery_mode:
        # only in non fast recovery mode, we can keep random transforms the same
        random_horizontal_flip = vision.RandomHorizontalFlip()
        random_color_adjust = vision.RandomColorAdjust()
        dataset = dataset.map([random_horizontal_flip, random_color_adjust])

    dataset = dataset.batch(4)

    num_epochs = 2
    dataset_size = dataset.get_dataset_size()
    expected_result = get_expected_result(dataset, num_epochs)
    # test initialize from the beginning, the middle of the first epoch,
    # the end of the first epoch and the middle of the second epoch
    for init_step in [0, dataset_size // 2, dataset_size, dataset_size + dataset_size // 2]:
        init_dataset_from_step_and_verify_result(dataset, init_step, num_epochs, dataset_size, expected_result)

    ds.config.set_fast_recovery(original_mode)
    ds.config.set_seed(original_seed)


@pytest.mark.parametrize("init_step", list(range(8)))
def test_getter(init_step):
    """
    Feature: Pipeline resuming
    Description: Test getter method in pipeline resuming
    Expectation: The result should be the same with normal pipeline
    """
    dataset = ds.ImageFolderDataset("../data/dataset/testPK/data")
    random_crop_decode_resize = vision.RandomCropDecodeResize((32, 32))
    dataset = dataset.map([random_crop_decode_resize], input_columns=["image"])
    dataset = dataset.skip(4)
    dataset = dataset.take(20)
    dataset = dataset.batch(5)
    dataset.set_init_step(init_step)

    assert dataset.get_dataset_size() == 4
    assert dataset.get_col_names() == ["image", "label"]
    assert dataset.output_shapes() == [[5, 32, 32, 3], [5]]
    assert dataset.output_types() == ["uint8", "int32"]
    assert dataset.num_classes() == 4
    assert dataset.get_batch_size() == 5
    assert dataset.get_repeat_count() == 1


if __name__ == "__main__":
    test_init_step_with_non_mappable_source(fast_recovery_mode=True, shuffle=True)
    test_init_step_with_mappable_source(fast_recovery_mode=True, shuffle=True)
    test_init_step_with_non_mappable_generator(fast_recovery_mode=True, shuffle=True)
    test_init_step_with_mappable_generator(fast_recovery_mode=True, shuffle=True)
    test_getter(init_step=0)
