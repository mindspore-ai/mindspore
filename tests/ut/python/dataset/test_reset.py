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
Testing dataset pipeline failover Reset
"""
import os
import numpy as np
import pytest
from util_minddataset import add_and_remove_cv_file  # pylint: disable=unused-import
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

# pylint: disable=no-value-for-parameter


def create_np_dataset(size):
    dimensions = (size, 4, 3, 2)
    np_data = np.random.random(dimensions)
    data = ds.NumpySlicesDataset(np_data, shuffle=False)
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














def test_reset_generator_with_unknow_length():
    """
    Feature: Failover
    Description: Reset unknown length GeneratorDataset
    Expectation: The recovery should not hang and can iterate as many times as expected
    """

    def gen():
        yield from range(5)

    dataset = ds.GeneratorDataset(gen, ["data"])
    iterator = dataset.create_dict_iterator(output_numpy=True, num_epochs=1)
    ds.engine.datasets._set_training_dataset(iterator)  # pylint: disable=W0212
    for i in range(5):
        ds.engine.datasets._reset_training_dataset(i, 5)  # pylint: disable=W0212
        count = 0
        for _ in enumerate(iterator):
            count += 1
        assert count == 5 - i
