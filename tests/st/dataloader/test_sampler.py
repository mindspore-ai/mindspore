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
"""Test Sampler."""

import numpy as np
import pytest

import mindspore as ms
from mindspore.dataset.dataloader import (
    BatchSampler,
    DataLoader,
    Dataset,
    IterableDataset,
    SequentialSampler,
    RandomSampler,
    DistributedSampler,
)

from tests.mark_utils import arg_mark


class MyDataset(Dataset):
    """
    A map style dataset that returns as many samples as requested.
    """

    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
        self.data = list(range(num_samples))

    def __getitem__(self, index):
        return np.array(self.data[index])

    def __len__(self):
        return self.num_samples


class MyIterDataset(IterableDataset):
    """
    An iterable style dataset that yields as many samples as requested.
    """

    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
        self.data = [np.array(idx) for idx in range(num_samples)]

    def __iter__(self):
        return iter(self.data)


class MySampler:
    """
    A sampler that yields as many indices as requested sequentially.
    """

    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.num_samples:
            data = self.index
            self.index += 1
            return data
        raise StopIteration


def compare_tensor_list(list1, list2):
    """
    Compare two lists of tensors.
    """
    assert len(list1) == len(list2)
    for v1, v2 in zip(list1, list2):
        assert (v1 == v2).all()


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_dataloader_udf_sampler():
    """
    Feature: Test DataLoader sampler.
    Description: Test the DataLoader with UDF sampler.
    Expectation: The result is as expected.
    """

    dataset = MyDataset(10)
    sampler = MySampler(6)

    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor([i, i + 1]) for i in range(0, 6, 2)]
    compare_tensor_list(result, expect)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_dataloader_sequential_sampler():
    """
    Feature: Test DataLoader sampler.
    Description: Test the DataLoader with SequentialSampler.
    Expectation: The result is as expected.
    """

    dataset = MyDataset(10)

    dataloader1 = DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset), shuffle=False)
    result1 = list(dataloader1)

    dataloader2 = DataLoader(dataset, batch_size=1, sampler=None, shuffle=False)
    result2 = list(dataloader2)

    compare_tensor_list(result1, result2)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_dataloader_random_sampler():
    """
    Feature: Test DataLoader sampler.
    Description: Test the DataLoader with RandomSampler.
    Expectation: The result is as expected.
    """

    dataset = MyDataset(10)

    generator = np.random.default_rng(40)
    expected_value = [
        ms.Tensor([5]),
        ms.Tensor([7]),
        ms.Tensor([0]),
        ms.Tensor([6]),
        ms.Tensor([4]),
        ms.Tensor([9]),
        ms.Tensor([0]),
        ms.Tensor([0]),
        ms.Tensor([4]),
        ms.Tensor([6]),
    ]
    sampler = RandomSampler(dataset, replacement=True, generator=generator)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=False)
    result = list(dataloader)
    assert result == expected_value

    expected_value = [ms.Tensor([0]), ms.Tensor([9]), ms.Tensor([7])]
    sampler = RandomSampler(dataset, replacement=True, num_samples=3, generator=generator)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=False)
    result = list(dataloader)
    assert result == expected_value

    expected_value = [
        ms.Tensor([2]),
        ms.Tensor([3]),
        ms.Tensor([5]),
        ms.Tensor([1]),
        ms.Tensor([6]),
        ms.Tensor([0]),
        ms.Tensor([8]),
        ms.Tensor([4]),
        ms.Tensor([7]),
        ms.Tensor([9]),
    ]
    sampler = RandomSampler(dataset, replacement=False, generator=generator)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=False)
    result = list(dataloader)
    assert result == expected_value


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_dataloader_batch_sampler():
    """
    Feature: Test DataLoader sampler.
    Description: Test the DataLoader with batch_sampler.
    Expectation: The result is as expected.
    """

    dataset = MyDataset(10)

    class SimpleBatchSampler:
        """
        A simple batch sampler that yields a batch of indices each time.
        """

        def __init__(self):
            self.indices = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

        def __iter__(self):
            return iter(self.indices)

    dataloader = DataLoader(dataset, batch_size=1, batch_sampler=SimpleBatchSampler(), shuffle=False)
    result = list(dataloader)
    expected = [ms.Tensor([i, i + 1]) for i in range(0, 10, 2)]
    compare_tensor_list(result, expected)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_dataloader_user_defined_sampler():
    """
    Feature: Test DataLoader sampler.
    Description: Test the DataLoader with user defined sampler.
    Expectation: The result is as expected.
    """

    dataset = MyDataset(10)
    sampler = MySampler(6)

    dataloader = DataLoader(dataset, sampler=sampler, shuffle=False)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in range(6)]
    compare_tensor_list(result, expect)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_dataloader_distributed_sampler():
    """
    Feature: Test DataLoader sampler.
    Description: Test the DataLoader with DistributedSampler.
    Expectation: The result is as expected.
    """

    dataset = MyDataset(10)

    sampler = DistributedSampler(dataset, shuffle=False, num_replicas=None, rank=None)
    dataloader = DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in range(10)]
    compare_tensor_list(result, expect)
    print(result)

    sampler = DistributedSampler(dataset, shuffle=False, num_replicas=2, rank=0)
    dataloader = DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in range(0, 10, 2)]
    compare_tensor_list(result, expect)
    print(result)

    sampler = DistributedSampler(dataset, shuffle=False, num_replicas=2, rank=1)
    dataloader = DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in range(1, 10, 2)]
    compare_tensor_list(result, expect)
    print(result)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_dataloader_distributed_sampler_shuffle():
    """
    Feature: Test DataLoader sampler.
    Description: Test the DataLoader with DistributedSampler and shuffle.
    Expectation: The result is as expected.
    """

    dataset = MyDataset(10)

    sampler = DistributedSampler(dataset, shuffle=True, seed=1, num_replicas=2, rank=0)
    dataloader = DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in [8, 7, 1, 5, 6]]
    compare_tensor_list(result, expect)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_dataloader_distributed_sampler_drop_last():
    """
    Feature: Test DataLoader sampler.
    Description: Test the DataLoader with DistributedSampler and drop_last.
    Expectation: The result is as expected.
    """

    dataset = MyDataset(10)

    sampler = DistributedSampler(dataset, shuffle=False, num_replicas=3, rank=2, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in [2, 5, 8, 1]]
    compare_tensor_list(result, expect)

    sampler = DistributedSampler(dataset, shuffle=False, num_replicas=3, rank=2, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in [2, 5, 8]]
    compare_tensor_list(result, expect)

    sampler = DistributedSampler(dataset, shuffle=False, num_replicas=20, rank=0, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in [2, 5, 8]]


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_sequential_sampler():
    """
    Feature: Sequential Sampler
    Description: Verify the functionality of the sequential sampler
    Expectation: Success
    """
    dataset = MyDataset(10)

    result = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sequential_sampler = SequentialSampler(dataset)
    assert list(sequential_sampler) == result


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_random_sampler():
    """
    Feature: Random Sampler
    Description: Verify the functionality of the random sampler
    Expectation: Success
    """
    generator = np.random.default_rng(40)

    dataset = MyDataset(10)

    # Default parameters
    result = [1, 7, 8, 5, 3, 4, 2, 0, 9, 6]
    random_sampler = RandomSampler(dataset, generator=generator)
    assert list(random_sampler) == result

    # replacement is True
    result_1 = [6, 1, 0, 5, 0, 8, 3, 6, 2, 1]
    random_sampler_1 = RandomSampler(dataset, replacement=True, generator=generator)
    assert list(random_sampler_1) == result_1

    # replacement is True and num_samples is 15
    result_2 = [1, 2, 0, 5, 3, 7, 4, 9, 8, 6, 0, 7, 6, 5, 2]
    random_sampler_2 = RandomSampler(dataset, replacement=False, num_samples=15, generator=generator)
    assert list(random_sampler_2) == result_2


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_dataloader_random_sampler_exception():
    """
    Feature: Random sampling of abnormal scenarios
    Description: Verify abnormal scenarios in random sampling
    Expectation: Success
    """
    dataset = MyDataset(10)

    # 1.Verify scenarios where the replacement parameter is not of type bool.
    error_msssage = "replacement must be bool, but got: int"
    with pytest.raises(TypeError) as error_info:
        replacement = 1
        _ = RandomSampler(dataset, replacement=replacement)
    assert error_msssage in str(error_info.value)

    # 2.Verify scenarios where the num_samples parameter is not of type int.
    error_msssage = "num_samples must be int, but got: str"
    with pytest.raises(TypeError) as error_info:
        num_samples = "test"
        _ = RandomSampler(dataset, num_samples=num_samples)
    assert error_msssage in str(error_info.value)

    # 3.Verify scenarios where the num_samples parameter is less than or equal to 0.
    error_msssage = "num_samples must be a positive integer value, but got num_samples = 0"
    with pytest.raises(ValueError) as error_info:
        num_samples = 0
        _ = RandomSampler(dataset, num_samples=num_samples)
    assert error_msssage in str(error_info.value)

    # 4.Verify that the generator parameter is not of type mindspore.Generator.
    error_msssage = "generator must be numpy.random.Generator, but got: int"
    with pytest.raises(TypeError) as error_info:
        generator = 0
        _ = RandomSampler(dataset, generator=generator)
    assert error_msssage in str(error_info.value)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_batch_sampler():
    """
    Feature: Batch Sampler
    Description: Verify the functionality of the batch sampler
    Expectation: Success
    """
    dataset = MyDataset(10)

    result = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size=3, drop_last=False)
    assert list(batch_sampler) == result

    result_1 = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    sampler_1 = SequentialSampler(dataset)
    batch_sampler_1 = BatchSampler(sampler_1, batch_size=3, drop_last=True)
    assert list(batch_sampler_1) == result_1

    result_2 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    sampler_2 = SequentialSampler(dataset)
    batch_sampler_2 = BatchSampler(sampler_2, batch_size=11, drop_last=False)
    assert list(batch_sampler_2) == result_2


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_dataloader_batch_sampler_exception():
    """
    Feature: Batch sampling of abnormal scenarios
    Description: Verify abnormal scenarios in batch sampling
    Expectation: Success
    """
    dataset = MyDataset(10)

    sampler = SequentialSampler(dataset)

    # 1.Verify that the batch_size parameter is an integer type.
    error_msssage = "batch_size must be <class 'int'>"
    with pytest.raises(TypeError) as error_info:
        batch_size = "test"
        _ = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    assert error_msssage in str(error_info.value)

    # 2.Verify that the batch_size parameter is less than or equal to 0.
    error_msssage = "batch_size must be positive"
    with pytest.raises(ValueError) as error_info:
        batch_size = 0
        _ = BatchSampler(sampler, batch_size, drop_last=False)
    assert error_msssage in str(error_info.value)

    # 3.Verify that the drop_last parameter is a boolean type.
    error_msssage = "drop_last must be <class 'bool'>"
    with pytest.raises(TypeError) as error_info:
        drop_last = 1
        _ = BatchSampler(sampler, batch_size=2, drop_last=drop_last)
    assert error_msssage in str(error_info.value)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_distributed_sampler():
    """
    Feature: Distribute Sampler
    Description: Verify the functionality of the distribute sampler
    Expectation: Success
    """
    dataset = MyDataset(10)

    result = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    distributed_sampler = DistributedSampler(dataset, shuffle=False, num_replicas=None, rank=None)
    assert list(distributed_sampler) == result

    result_1 = [0, 3, 6, 9]
    distributed_sampler_1 = DistributedSampler(dataset, shuffle=False, num_replicas=3, rank=0)
    assert list(distributed_sampler_1) == result_1

    result_2 = [0, 3, 6]
    distributed_sampler_2 = DistributedSampler(dataset, shuffle=False, num_replicas=3, rank=0, drop_last=True)
    assert list(distributed_sampler_2) == result_2

    result_3 = [1, 7, 8, 5, 3, 4, 2, 0, 9, 6]
    distributed_sampler_3 = DistributedSampler(dataset, shuffle=True, seed=40)
    assert list(distributed_sampler_3) == result_3


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_dataloader_distributed_sampler_exception():
    """
    Feature: Distribute sampling of abnormal scenarios
    Description: Verify abnormal scenarios in distribute sampling
    Expectation: Success
    """
    dataset = MyDataset(10)

    # 1.Verify that the num_replicas parameter is not an integer type.
    error_msssage = "num_replicas must be int, but got: str"
    with pytest.raises(TypeError) as error_info:
        num_replicas = "test"
        _ = DistributedSampler(dataset, num_replicas=num_replicas)
    assert error_msssage in str(error_info.value)

    # 2.Verify that the rank parameter is not an integer type.
    error_msssage = "rank must be int, but got: str"
    with pytest.raises(TypeError) as error_info:
        rank = "test"
        _ = DistributedSampler(dataset, rank=rank)
    assert error_msssage in str(error_info.value)

    # 3.Verify that the seed parameter is not an integer type.
    error_msssage = "seed must be int, but got: str"
    with pytest.raises(TypeError) as error_info:
        seed = "test"
        _ = DistributedSampler(dataset, seed=seed)
    assert error_msssage in str(error_info.value)

    # 4.Verify that the shuffle parameter is not an boolean type.
    error_msssage = "shuffle must be bool, but got: str"
    with pytest.raises(TypeError) as error_info:
        shuffle = "test"
        _ = DistributedSampler(dataset, shuffle=shuffle)
    assert error_msssage in str(error_info.value)

    # 5.Verify that the drop_last parameter is not an boolean type.
    error_msssage = "drop_last must be bool, but got: str"
    with pytest.raises(TypeError) as error_info:
        drop_last = "test"
        _ = DistributedSampler(dataset, drop_last=drop_last)
    assert error_msssage in str(error_info.value)

    # 6.Verify scenarios where the num_replicas parameter is less than or equal to 0.
    error_msssage = "Invalid num_replicas: 0, num_replicas must be greater than 0"
    with pytest.raises(ValueError) as error_info:
        num_replicas = 0
        _ = DistributedSampler(dataset, num_replicas=num_replicas)
    assert error_msssage in str(error_info.value)

    # 7.Verify that the rank parameter is less than 0 or not in the range [0, num_replicas-1].
    error_msssage = "Invalid rank: 5, rank must be in the interval [0, 3]"
    with pytest.raises(ValueError) as error_info:
        rank = 5
        _ = DistributedSampler(dataset, num_replicas=4, rank=rank)
    assert error_msssage in str(error_info.value)

    error_msssage = "Invalid rank: -1, rank must be in the interval [0, 3]"
    with pytest.raises(ValueError) as error_info:
        rank = -1
        _ = DistributedSampler(dataset, num_replicas=4, rank=rank)
    assert error_msssage in str(error_info.value)
