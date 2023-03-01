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

# Need to run all these tests in separate processes since MD internally stores
# "training" dataset in a global variable every time.
pytestmark = pytest.mark.forked


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


def create_random_imagenet_dataset(repeat_size, sampler=None, num_parallel_workers=1, to_pil=False, batch_func=None):
    shuffle = True if sampler is None else None
    data_dir = "../data/dataset/testImageNetData2/train"
    data = ds.ImageFolderDataset(
        data_dir, shuffle=shuffle, sampler=sampler)
    data = data.repeat(repeat_size)
    crop_op1 = vision.RandomCrop(4)
    operations = [vision.Decode(to_pil=to_pil), crop_op1]
    if to_pil:  # include a pyfunc in test if to_pil is True
        operations.append(lambda x: x.rotate(45))
    data = data.map(operations=operations, input_columns=[
        "image"], num_parallel_workers=num_parallel_workers, python_multiprocessing=True)
    if batch_func:
        data = data.batch(batch_size=2, per_batch_map=batch_func, input_columns=["label"],
                          num_parallel_workers=num_parallel_workers, python_multiprocessing=True)
    data = data.project(["image"])
    return data


def create_minddata_dataset(size):
    columns_list = ["data"]
    num_readers = 2
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data = ds.MindDataset(file_name + "0", columns_list, num_readers, shuffle=False, num_samples=size)
    data = data.rename(input_columns=["data"], output_columns="fake_data")
    return data


def run_reset(data, num_epochs: int, failure_point: int):
    size = data.get_dataset_size()
    expected = []
    expected_itr = data.create_tuple_iterator(num_epochs=num_epochs, output_numpy=True)
    for _ in range(num_epochs):
        for d in expected_itr:
            expected.append(d)
    del expected_itr

    expected2 = []
    itr = data.create_tuple_iterator(num_epochs=num_epochs, output_numpy=True)
    ds.engine.datasets._set_training_dataset(itr)  # pylint: disable=W0212
    cur_step: int = 0
    failed = False
    for _ in range(num_epochs):
        for d in itr:
            expected2.append(d)
            if cur_step + 1 == failure_point:
                # pylint: disable=W0212
                ds.engine.datasets._reset_training_dataset(failure_point, size)
                failed = True
                break
            cur_step += 1
        if failed:
            break

    if failed:
        for _ in range(failure_point // size, num_epochs):
            for d in itr:
                expected2.append(d)

    with pytest.raises(RuntimeError, match="User tries to fetch data beyond the specified number of epochs."):
        for _ in itr:
            expected2.append(d)

    assert len(expected) == len(expected2)
    for x, y in zip(expected, expected2):
        np.testing.assert_array_equal(x, y)


def run_reset_error(data, num_epochs: int, failure_point: int):
    itr = data.create_tuple_iterator(num_epochs=num_epochs, output_numpy=True)  # pylint: disable=unused-variable
    ds.engine.datasets._set_training_dataset(itr)  # pylint: disable=W0212
    dataset_size = data.get_dataset_size()

    if failure_point > 0:
        with pytest.raises(RuntimeError) as err:
            # pylint: disable=W0212
            ds.engine.datasets._reset_training_dataset(failure_point, dataset_size)
        assert "Cannot reset the pipeline, reset step must be less than dataset_size * num_epochs." in str(err.value)
    else:
        with pytest.raises(RuntimeError) as err:
            # pylint: disable=W0212
            ds.engine.datasets._reset_training_dataset(failure_point, dataset_size)
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
        run_reset(data, num_epochs=num_epochs, failure_point=failure_point)


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
        run_reset(data, num_epochs=num_epochs, failure_point=failure_point)


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
        run_reset(data, num_epochs=num_epochs, failure_point=failure_point)


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
        run_reset(data, num_epochs=num_epochs, failure_point=failure_point)


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
        run_reset(data, num_epochs=num_epochs, failure_point=failure_point)


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


def random_col(col1, batch_info):
    return ([np.random.rand(1) for a in col1],)


@pytest.mark.parametrize("num_parallel_workers", (4, 5))
@pytest.mark.parametrize("sampler", (ds.RandomSampler(), None))
@pytest.mark.parametrize("to_pil, batch_func", [(False, None), (True, random_col)])  # test C ops and Python ops (MP)
def test_repeatable_reset_imagenet(sampler, num_parallel_workers, to_pil, batch_func):
    """
    Feature: Dataset recovery
    Description: Simple test of data pipeline with fast_recovery set to False
    Expectation: Same dataset after reset
    """
    num_epochs = 4
    original_seed = ds.config.get_seed()
    original_fast_recovery = ds.config.get_fast_recovery()
    original_shared_mem = ds.config.get_enable_shared_mem()
    ds.config.set_seed(100)
    ds.config.set_fast_recovery(False)
    ds.config.set_enable_shared_mem(False)

    expected = []
    data = create_random_imagenet_dataset(
        repeat_size=1, sampler=sampler, to_pil=to_pil, num_parallel_workers=num_parallel_workers, batch_func=batch_func)
    expected_itr = data.create_tuple_iterator(
        num_epochs=num_epochs, output_numpy=True)

    # successful run (to collect correct output)
    for _ in range(num_epochs):
        for d in expected_itr:
            expected.append(d)
    del expected_itr
    dataset_size = data.get_dataset_size()
    # try different failure points
    for failure_point in (5, 6, 22):
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
                # pylint: disable=W0212
                ds.engine.datasets._reset_training_dataset(failure_point, dataset_size)
                failure = False
                for d in expected2_itr:
                    expected2.append(d)
        del expected2_itr

        # verify count and values of failover with original run
        np.testing.assert_array_equal(expected, expected2)

    ds.config.set_seed(original_seed)
    ds.config.set_fast_recovery(original_fast_recovery)
    ds.config.set_enable_shared_mem(original_shared_mem)


@pytest.mark.parametrize("to_pil", (False, True))  # test C ops and Python ops with MP=true
@pytest.mark.parametrize("num_parallel_workers", (4, 5))
@pytest.mark.parametrize("shard_id", (0, 1, 2, 3))
def test_repeatable_reset_distributed(shard_id, num_parallel_workers, to_pil):
    """
    Feature: Dataset recovery
    Description: Simple test of data pipeline with fast_recovery set to False for a distributed sampler
    Expectation: Same dataset after reset
    """
    num_shards = 4
    num_epochs = 3
    original_seed = ds.config.get_seed()
    original_fast_recovery = ds.config.get_fast_recovery()
    original_shared_mem = ds.config.get_enable_shared_mem()
    ds.config.set_seed(100)
    ds.config.set_fast_recovery(False)
    ds.config.set_enable_shared_mem(False)

    expected = []
    distributed_sampler = ds.DistributedSampler(
        num_shards=num_shards, shard_id=shard_id)
    data = create_random_imagenet_dataset(
        repeat_size=2, sampler=distributed_sampler, num_parallel_workers=num_parallel_workers, to_pil=to_pil)
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
    for failure_point in (3, 7, 9):
        expected2 = []
        expected2_itr = data.create_tuple_iterator(num_epochs=num_epochs, output_numpy=True)
        ds.engine.datasets._set_training_dataset(expected2_itr)  # pylint: disable=W0212
        failure = False
        for epoch in range(num_epochs):
            for step, d in enumerate(expected2_itr):
                expected2.append(d)
                if epoch * data.get_dataset_size() + step + 1 == failure_point:
                    failure = True
                    break
            if failure:
                # pylint: disable=W0212
                ds.engine.datasets._reset_training_dataset(failure_point, data.get_dataset_size())
                failure = False
                for d in expected2_itr:
                    expected2.append(d)

        # verify count and values of failover with original run
        np.testing.assert_array_equal(expected, expected2)

    ds.config.set_seed(original_seed)
    ds.config.set_fast_recovery(original_fast_recovery)
    ds.config.set_enable_shared_mem(original_shared_mem)


def test_reset_shuffle():
    """
    Feature: Dataset recovery
    Description: The random generator in shuffle operation resets to correct internal state
    Expectation: Same dataset after reset
    """
    original_seed = ds.config.get_seed()
    original_fast_recovery = ds.config.get_fast_recovery()
    ds.config.set_seed(1)
    ds.config.set_fast_recovery(True)

    source = [(np.array([x])) for x in range(10)]
    data1 = ds.NumpySlicesDataset(source, ["data"], sampler=ds.SequentialSampler())
    data1 = data1.shuffle(3)
    data1 = data1.skip(1)
    num_epochs = 3

    expected = []
    expected_itr = data1.create_tuple_iterator(num_epochs=num_epochs, output_numpy=True)
    for epoch in range(num_epochs):
        for step, d in enumerate(expected_itr):
            expected.append(d)

    failure_point = 13
    expected2 = []
    expected2_itr = data1.create_tuple_iterator(num_epochs=num_epochs, output_numpy=True)
    ds.engine.datasets._set_training_dataset(expected2_itr)  # pylint: disable=W0212
    failure = False
    for epoch in range(num_epochs):
        for step, d in enumerate(expected2_itr):
            expected2.append(d)
            if epoch * data1.get_dataset_size() + step + 1 == failure_point:
                failure = True
                break
        if failure:
            ds.engine.datasets._reset_training_dataset(failure_point, data1.get_dataset_size())  # pylint: disable=W0212
            failure = False
            for step, d in enumerate(expected2_itr):
                expected2.append(d)

    with pytest.raises(RuntimeError, match="User tries to fetch data beyond the specified number of epochs."):
        for step, d in enumerate(expected2_itr):
            expected2.append(d)
    np.testing.assert_array_equal(expected, expected2)

    ds.config.set_seed(original_seed)
    ds.config.set_fast_recovery(original_fast_recovery)


@pytest.mark.parametrize("sampler", (ds.RandomSampler(), ds.SequentialSampler()))
def test_reset_sampler(sampler):
    """
    Feature: Dataset recovery
    Description: The samplers for source operations reset to correct internal state.
    Expectation: Same dataset after reset
    """
    original_seed = ds.config.get_seed()
    original_fast_recovery = ds.config.get_fast_recovery()
    ds.config.set_seed(1)
    ds.config.set_fast_recovery(True)

    source = [(np.array([x]),) for x in range(10)]
    data1 = ds.NumpySlicesDataset(source, ["data"], sampler=sampler)
    data1 = data1.skip(1)
    data1 = data1.repeat(2)
    data1 = data1.skip(1)
    num_epochs = 3

    expected_itr = data1.create_tuple_iterator(
        num_epochs=num_epochs, output_numpy=True)
    expected = []
    for epoch in range(num_epochs):
        for step, d in enumerate(expected_itr):
            expected.append(d)

    failure_point = 13
    expected2 = []
    expected2_itr = data1.create_tuple_iterator(num_epochs=num_epochs, output_numpy=True)
    ds.engine.datasets._set_training_dataset(expected2_itr)  # pylint: disable=W0212
    failure = False

    for epoch in range(num_epochs):
        for step, d in enumerate(expected2_itr):
            expected2.append(d)
            if epoch * data1.get_dataset_size() + step + 1 == failure_point:
                failure = True
                break
        if failure:
            ds.engine.datasets._reset_training_dataset(failure_point, data1.get_dataset_size())  # pylint: disable=W0212
            failure = False
            for step, d in enumerate(expected2_itr):
                expected2.append(d)

    with pytest.raises(RuntimeError, match="User tries to fetch data beyond the specified number of epochs."):
        for step, d in enumerate(expected2_itr):
            expected2.append(d)
    np.testing.assert_array_equal(expected, expected2)

    ds.config.set_seed(original_seed)
    ds.config.set_fast_recovery(original_fast_recovery)


@pytest.mark.parametrize("fast_recovery", (False, True))
def test_reset_batch(fast_recovery):
    """
    Feature: Dataset recovery
    Description: The BatchInfo argument of batch operation contains correct information  (epoch num)
    Expectation: Test succeeds
    """
    original_fast_recovery = ds.config.get_fast_recovery()
    ds.config.set_fast_recovery(fast_recovery)

    num_epochs = 5
    repeat_size = 4
    skip_size = 12

    def get_epoch_num(col1, batch_info):
        return (np.array(batch_info.get_epoch_num()),)

    data1 = ds.NumpySlicesDataset(np.arange(10).reshape(10, 1))
    data1 = data1.repeat(repeat_size)
    data1 = data1.skip(skip_size)
    data1 = data1.batch(batch_size=7, per_batch_map=get_epoch_num, num_parallel_workers=1, python_multiprocessing=False)

    itr = data1.create_tuple_iterator(num_epochs=num_epochs, output_numpy=True)
    ds.engine.datasets._set_training_dataset(itr)  # pylint: disable=W0212

    failure = False
    failure_point = 25
    expected = np.repeat(np.arange(5), 4).reshape((20, 1))
    expected2 = []

    for epoch in range(num_epochs):
        for step, d in enumerate(itr):
            expected2.append(d)
            if epoch * data1.get_dataset_size() + step + 1 == failure_point:
                failure = True
                break
        if failure:
            ds.engine.datasets._reset_training_dataset(failure_point, data1.get_dataset_size())  # pylint: disable=W0212
            failure = False
            for step, d in enumerate(itr):
                expected2.append(d)

    with pytest.raises(RuntimeError, match="User tries to fetch data beyond the specified number of epochs."):
        for d in itr:
            expected2.append(d)
    np.testing.assert_array_equal(expected, expected2)

    ds.config.set_fast_recovery(original_fast_recovery)


def test_reset_nonmappable():
    """
    Feature: Dataset recovery
    Description: The order of rows read in normal and reset runs are identical for a TFRecord dataset.
    Expectation: Test succeeds
    """
    original_seed = ds.config.get_seed()
    original_fast_recovery = ds.config.get_fast_recovery()

    num_epochs = 10
    num_repeats = 5
    tf_files = ["../data/dataset/tf_file_dataset/test1.data", "../data/dataset/tf_file_dataset/test2.data",
                "../data/dataset/tf_file_dataset/test3.data", "../data/dataset/tf_file_dataset/test4.data"]

    # run a pipeline and collect rows
    def get_res(shard_id, num_repeats, failure_point):
        ds.config.set_seed(1)
        ds.config.set_fast_recovery(True)

        data1 = ds.TFRecordDataset(tf_files, num_shards=4, shard_id=shard_id, num_samples=5, shuffle=ds.Shuffle.FILES)
        data1 = data1.repeat(num_repeats)
        itr = data1.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)
        ds.engine.datasets._set_training_dataset(itr)  # pylint: disable=W0212
        dataset_size = data1.get_dataset_size()

        res = list()
        failure = False
        for epoch in range(num_epochs):
            for step, item in enumerate(itr):
                res.append(item["scalars"][0])
                if epoch * dataset_size + step + 1 == failure_point:
                    failure = True
                    break
            if failure:
                # pylint: disable=W0212
                ds.engine.datasets._reset_training_dataset(failure_point, dataset_size)
                failure = False
                # let's collect the remaining rows of this epoch
                if failure_point % dataset_size != 0:
                    for step, item in enumerate(itr):
                        res.append(item["scalars"][0])
        return res

    shard_id = 0
    expected = get_res(0, 5, -1) # no reset in this run
    # try different failure points and compare against 'expected'
    for failure_point in range(100):
        expected2 = get_res(shard_id, num_repeats, failure_point)
        np.testing.assert_array_equal(expected, expected2)

    ds.config.set_seed(original_seed)
    ds.config.set_fast_recovery(original_fast_recovery)


if __name__ == "__main__":
    test_reset_np()
    test_reset_cifar1()
    test_reset_cifar2()
    test_reset_imagenet()
    test_reset_mindrecord(add_and_remove_cv_file)
    test_reset_np_error()
    test_repeatable_reset_imagenet(3, None, False, None)
    test_repeatable_reset_distributed(1, 2, True)
    test_reset_shuffle()
    test_reset_sampler(ds.RandomSampler())
    test_reset_batch(False)
    test_reset_nonmappable()
