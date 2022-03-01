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
Testing pipeline Reset
"""
import numpy as np
import pytest
import mindspore.dataset as ds


def create_np_dataset(size):
    data = ds.NumpySlicesDataset(list(range(1, size + 1)), shuffle=False)
    return data


def util(data, num_epochs, failure_point: int, reset_step):
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


def test_reset():
    """
    Feature: dataset recovery
    Description: Simple test of data pipeline reset feature
    Expectation: same datasets after reset
    """
    dataset_size = 5
    num_epochs = 3
    data = create_np_dataset(size=dataset_size)
    for failure_point in range(dataset_size * num_epochs):
        for reset_step in range(dataset_size * num_epochs):
            util(data, num_epochs=num_epochs, failure_point=failure_point, reset_step=reset_step)


if __name__ == "__main__":
    test_reset()
