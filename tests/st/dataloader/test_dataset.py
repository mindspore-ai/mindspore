# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test Dataset."""

import numpy as np
import pytest

import mindspore as ms
from mindspore.dataset.dataloader import Dataset, IterableDataset, TensorDataset
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_iterate_dataset():
    """
    Feature: Test Dataset.
    Description: Test iterate the Dataset.
    Expectation: Raise NotImplementedError.
    """

    dataset = Dataset()
    with pytest.raises(NotImplementedError, match="Dataset must implement __getitem__ method"):
        for _ in dataset:
            pass


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_len_dataset():
    """
    Feature: Test Dataset.
    Description: Test calculate the length of the Dataset.
    Expectation: Raise NotImplementedError.
    """

    dataset = Dataset()
    with pytest.raises(TypeError, match="object of type 'Dataset' has no len()"):
        _ = len(dataset)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_iterate_iterable_dataset():
    """
    Feature: Test IterableDataset.
    Description: Test iterate the IterableDataset.
    Expectation: Raise NotImplementedError.
    """

    dataset = IterableDataset()
    with pytest.raises(NotImplementedError, match="IterableDataset must implement __iter__ method"):
        for _ in dataset:
            pass


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_len_iterable_dataset():
    """
    Feature: Test IterableDataset.
    Description: Test calculate the length of the IterableDataset.
    Expectation: Raise NotImplementedError.
    """

    dataset = IterableDataset()
    with pytest.raises(TypeError, match="object of type 'IterableDataset' has no len()"):
        _ = len(dataset)


class TestTensorDataset:
    """Class for testing TensorDataset."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_getitem(self):
        """
        Feature: Test TensorDataset.
        Description: Test the iteration of the TensorDataset.
        Expectation: The result is as expected.
        """
        images = np.random.randint(0, 255, (10, 28, 28))
        labels = np.random.randint(0, 10, (10,))
        dataset = TensorDataset(ms.Tensor(images), ms.Tensor(labels))
        for i, sample in enumerate(dataset):
            np.testing.assert_array_equal(sample[0].asnumpy(), images[i])
            np.testing.assert_array_equal(sample[1].asnumpy(), labels[i])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_len(self):
        """
        Feature: Test TensorDataset.
        Description: Test the length of the TensorDataset.
        Expectation: The result is as expected.
        """
        dataset = TensorDataset(ms.Tensor([0, 1, 2]), ms.Tensor([3, 4, 5]))
        assert len(dataset) == 3

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_invalid_tensors(self):
        """
        Feature: Test TensorDataset.
        Description: Test the invalid tensors.
        Expectation: Raise ValueError.
        """
        with pytest.raises(ValueError, match="All tensors must have the same size in the first dimension."):
            TensorDataset(ms.Tensor([0, 1, 2]), ms.Tensor([3, 4, 5, 6]))
