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
Testing dataset pipeline failover Reset
"""
import os
import numpy as np
import pytest
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from util_minddataset import add_and_remove_cv_file

# pylint: disable=no-value-for-parameter


def create_np_dataset(size):
    dimensions = (size, 4, 3, 2)
    np_data = np.random.random(dimensions)
    data = ds.NumpySlicesDataset(np_data, shuffle=False)
    return data


def create_cifar_dataset1(size):
    data_dir = "../data/dataset/testCifar100Data"
    pad_size = 100
    crop_size = 64
    data = ds.Cifar100Dataset(data_dir, num_samples=size, shuffle=False)
    data = data.project(["image"])
    pad_op = vision.Pad(pad_size)
    data = data.map(operations=pad_op, input_columns=["image"])
    crop_op = vision.CenterCrop(crop_size)
    data = data.map(operations=crop_op, input_columns=["image"])
    return data


def create_cifar_dataset2(size):
    data_dir = "../data/dataset/testCifar100Data"
    pad_size = 100
    crop_size = 64
    repeat_count = 2
    data = ds.Cifar100Dataset(data_dir, num_samples=size, shuffle=False)
    data = data.repeat(repeat_count)
    data = data.project(["image"])
    pad_op = vision.Pad(pad_size)
    data = data.map(operations=pad_op, input_columns=["image"])
    crop_op = vision.CenterCrop(crop_size)
    data = data.map(operations=crop_op, input_columns=["image"])
    return data


def create_imagenet_dataset(size):
    data_dir = "../data/dataset/testImageNetData2/train"
    batch_size = 2
    data = ds.ImageFolderDataset(data_dir, num_samples=size * batch_size, shuffle=False)
    data = data.batch(batch_size)
    data = data.project(["image"])
    return data


def create_random_imagenet_dataset(repeat_size, sampler=None, num_parallel_workers=1, to_pil=False):
    shuffle = True if sampler is None else None
    data_dir = "../data/dataset/testImageNetData2/train"
    data = ds.ImageFolderDataset(
        data_dir, shuffle=shuffle, sampler=sampler)
    data = data.repeat(repeat_size)
    crop_op1 = vision.RandomCrop(4)
    data = data.map(operations=[vision.Decode(to_pil=to_pil), crop_op1], input_columns=[
        "image"], num_parallel_workers=num_parallel_workers, python_multiprocessing=True)
    data = data.project(["image"])
    return data


def create_minddata_dataset(size):
    columns_list = ["data"]
    num_readers = 2
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data = ds.MindDataset(file_name + "0", columns_list, num_readers, shuffle=False, num_samples=size)
    data = data.rename(input_columns=["data"], output_columns="fake_data")
    return data


def run_reset(data, num_epochs, failure_point: int, reset_step: int):
    size = data.get_dataset_size()
    expected = []
    expected_itr = data.create_tuple_iterator(num_epochs=num_epochs, output_numpy=True)
    for _ in range(num_epochs):
        for d in expected_itr:
            expected.append(d)
    del expected_itr

    actual_before_reset = []
    itr = data.create_tuple_iterator(num_epochs=num_epochs, output_numpy=True)
    ds.engine.datasets._set_training_dataset(itr)  # pylint: disable=W0212
    cur_step: int = 0
    failed = False
    for _ in range(num_epochs):
        for d in itr:
            actual_before_reset.append(d)
            if cur_step == failure_point:
                ds.engine.datasets._reset_training_dataset(reset_step)  # pylint: disable=W0212
                failed = True
                break
            cur_step += 1
        if failed:
            break

    actual_after_reset = []
    if failed:
        for _ in range(reset_step // size, num_epochs):
            for d in itr:
                actual_after_reset.append(d)

    with pytest.raises(RuntimeError, match="User tries to fetch data beyond the specified number of epochs."):
        for _ in itr:
            pass

    for x, y in zip(expected[:failure_point], actual_before_reset):
        np.testing.assert_array_equal(x, y)

    for x, y in zip(expected[reset_step:], actual_after_reset):
        np.testing.assert_array_equal(x, y)


def run_reset_error(data, num_epochs: int, failure_point: int):
    itr = data.create_tuple_iterator(num_epochs=num_epochs, output_numpy=True)  # pylint: disable=unused-variable
    ds.engine.datasets._set_training_dataset(itr)  # pylint: disable=W0212

    if failure_point > 0:
        with pytest.raises(RuntimeError) as err:
            ds.engine.datasets._reset_training_dataset(failure_point)  # pylint: disable=W0212
        assert "Cannot reset the pipeline, reset step must be less than dataset_size * num_epochs." in str(err.value)
    else:
        with pytest.raises(RuntimeError) as err:
            ds.engine.datasets._reset_training_dataset(failure_point)  # pylint: disable=W0212
        assert "Cannot reset the pipeline, reset step must be >= 0." in str(err.value)


def test_reset_np():
    """
    Feature: Dataset recovery
    Description: Simple test of data pipeline reset feature on a pipeline with NumpySlicesDataset as a leaf node
    Expectation: Same datasets after reset
    """
    dataset_size = 50
    num_epochs = 3
    failure_steps = (dataset_size * num_epochs) // 10
    data = create_np_dataset(size=dataset_size)
    for failure_point in range(0, dataset_size * num_epochs, failure_steps):
        for reset_step in range(0, dataset_size * num_epochs, failure_steps):
            run_reset(data, num_epochs=num_epochs, failure_point=failure_point, reset_step=reset_step)


def test_reset_cifar1():
    """
    Feature: Dataset recovery
    Description: Simple test of data pipeline reset feature on a pipeline with Cifar100Dataset as a leaf node (1)
    Expectation: Same datasets after reset
    """
    dataset_size = 30
    num_epochs = 2
    failure_steps = (dataset_size * num_epochs) // 5
    data = create_cifar_dataset1(size=dataset_size)
    for failure_point in range(0, dataset_size * num_epochs, failure_steps):
        for reset_step in range(0, dataset_size * num_epochs, failure_steps):
            run_reset(data, num_epochs=num_epochs, failure_point=failure_point, reset_step=reset_step)


def test_reset_cifar2():
    """
    Feature: Dataset recovery
    Description: Simple test of data pipeline reset feature on a pipeline with Cifar100Dataset as a leaf node (2)
    Expectation: Same datasets after reset
    """
    dataset_size = 30
    num_epochs = 3
    failure_steps = (dataset_size * num_epochs) // 5
    data = create_cifar_dataset2(size=dataset_size)
    for failure_point in range(0, dataset_size * num_epochs, failure_steps):
        for reset_step in range(0, dataset_size * num_epochs, failure_steps):
            run_reset(data, num_epochs=num_epochs, failure_point=failure_point, reset_step=reset_step)


def test_reset_imagenet():
    """
    Feature: Dataset recovery
    Description: Simple test of data pipeline reset feature on a pipeline with ImageFolderDataset as a leaf node
    Expectation: Same datasets after reset
    """
    dataset_size = 3
    num_epochs = 4
    failure_steps = (dataset_size * num_epochs) // 4
    data = create_imagenet_dataset(size=dataset_size)
    for failure_point in range(0, dataset_size * num_epochs, failure_steps):
        for reset_step in range(0, dataset_size * num_epochs, failure_steps):
            run_reset(data, num_epochs=num_epochs, failure_point=failure_point, reset_step=reset_step)


def test_reset_mindrecord(add_and_remove_cv_file):  # pylint: disable=unused-argument, redefined-outer-name
    """
    Feature: Dataset recovery
    Description: Simple test of data pipeline reset feature on a pipeline with MindDataset as a leaf node
    Expectation: Same datasets after reset
    """
    dataset_size = 10
    num_epochs = 3
    failure_steps = (dataset_size * num_epochs) // 10
    data = create_minddata_dataset(size=dataset_size)
    for failure_point in range(0, dataset_size * num_epochs, failure_steps):
        for reset_step in range(0, dataset_size * num_epochs, failure_steps):
            run_reset(data, num_epochs=num_epochs, failure_point=failure_point, reset_step=reset_step)


def test_reset_np_error():
    """
    Feature: Dataset recovery
    Description: Simple test of data pipeline reset feature for error cases (step is negative, or larger than expected)
    Expectation: Failures are detected properly and correct error message is produced
    """
    dataset_size = 100
    num_epochs = 3
    failure_points = (-1000, -300, -99, -5, 300, 301, 1000)
    data = create_np_dataset(size=dataset_size)
    for failure_point in failure_points:
        run_reset_error(data, num_epochs=num_epochs, failure_point=failure_point)


@pytest.mark.parametrize("num_parallel_workers", (1, 4))
@pytest.mark.parametrize("sampler", (ds.RandomSampler(), None))
def test_repeatable_reset_imagenet(sampler, num_parallel_workers, to_pil=False):
    """
    Feature: Dataset recovery
    Description: Simple test of data pipeline with fast_recovery set to False
    Expectation: Same dataset after reset
    """
    num_epochs = 4
    original_seed = ds.config.get_seed()
    original_fast_recovery = ds.config.get_fast_recovery()
    ds.config.set_seed(100)
    ds.config.set_fast_recovery(False)

    expected = []
    data = create_random_imagenet_dataset(
        repeat_size=1, sampler=sampler, to_pil=to_pil, num_parallel_workers=num_parallel_workers)
    expected_itr = data.create_tuple_iterator(
        num_epochs=num_epochs, output_numpy=True)

    # successful run (to collect correct output)
    for _ in range(num_epochs):
        for d in expected_itr:
            expected.append(d)
    del expected_itr
    import mindspore.log as logger
    dataset_size = data.get_dataset_size()
    logger.error(dataset_size)
    # try different failure points
    for failure_point in range(1, dataset_size * num_epochs, 2):
        expected2 = []
        expected2_itr = data.create_tuple_iterator(
            num_epochs=num_epochs, output_numpy=True)
        ds.engine.datasets._set_training_dataset(expected2_itr)  # pylint: disable=W0212
        failure = False

        for epoch in range(num_epochs):
            for step, d in enumerate(expected2_itr):
                expected2.append(d)
                if epoch * dataset_size + step + 1 == failure_point:
                    failure = True
                    break
            if failure:
                ds.engine.datasets._reset_training_dataset(failure_point)  # pylint: disable=W0212
                failure = False
                for d in expected2_itr:
                    expected2.append(d)
        del expected2_itr

        # verify count and values of failover with original run
        assert len(expected) == len(expected2)
        for a, b in zip(expected, expected2):
            assert np.array_equal(a[0], b[0])

    ds.config.set_seed(original_seed)
    ds.config.set_fast_recovery(original_fast_recovery)


@pytest.mark.parametrize("num_parallel_workers", (1, 4))
def test_repeatable_reset_distributed(num_parallel_workers):
    """
    Feature: Dataset recovery
    Description: Simple test of data pipeline with fast_recovery set to False for a distributed sampler
    Expectation: Same dataset after reset
    """
    num_shards = 4
    num_epochs = 3
    original_seed = ds.config.get_seed()
    original_fast_recovery = ds.config.get_fast_recovery()
    ds.config.set_seed(100)
    ds.config.set_fast_recovery(False)

    for i in range(num_shards):
        expected = []
        distributed_sampler = ds.DistributedSampler(
            num_shards=num_shards, shard_id=i)
        data = create_random_imagenet_dataset(
            repeat_size=4, sampler=distributed_sampler, num_parallel_workers=num_parallel_workers)
        iter_counter = 0

        # successful run (to collect correct output)
        expected_itr = data.create_tuple_iterator(
            num_epochs=num_epochs, output_numpy=True)
        for _ in range(num_epochs):
            for d in expected_itr:
                expected.append(d)
                iter_counter += 1
        assert iter_counter == data.get_dataset_size() * num_epochs
        del expected_itr

        # try different failure points
        for failure_point in range(1, data.get_dataset_size() * num_epochs, 3):
            expected2 = []
            expected2_itr = data.create_tuple_iterator(
                num_epochs=num_epochs, output_numpy=True)
            ds.engine.datasets._set_training_dataset(expected2_itr)  # pylint: disable=W0212
            failure = False
            for epoch in range(num_epochs):
                for step, d in enumerate(expected2_itr):
                    expected2.append(d)
                    if epoch * data.get_dataset_size() + step + 1 == failure_point:
                        failure = True
                        break
                if failure:
                    ds.engine.datasets._reset_training_dataset(failure_point)  # pylint: disable=W0212
                    failure = False
                    for d in expected2_itr:
                        expected2.append(d)

            # verify count and values of failover with original run
            assert len(expected) == len(expected2)
            for a, b in zip(expected, expected2):
                assert np.array_equal(a, b)

    ds.config.set_seed(original_seed)
    ds.config.set_fast_recovery(original_fast_recovery)


if __name__ == "__main__":
    test_reset_np()
    test_reset_cifar1()
    test_reset_cifar2()
    test_reset_imagenet()
    test_reset_mindrecord(add_and_remove_cv_file)
    test_reset_np_error()
    test_repeatable_reset_imagenet()
    test_repeatable_reset_distributed()
