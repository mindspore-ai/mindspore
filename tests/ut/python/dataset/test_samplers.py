# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from mindspore import log as logger
from util import dataset_equal, config_get_set_seed


# test5trainimgs.json contains 5 images whose un-decoded shape is [83554, 54214, 65512, 54214, 64631]
# the label of each image is [0,0,0,1,1] each image can be uniquely identified
# via the following lookup table (dict){(83554, 0): 0, (54214, 0): 1, (54214, 1): 2, (65512, 0): 3, (64631, 1): 4}

def test_sequential_sampler(print_res=False):
    """
    Feature: SequentialSampler op
    Description: Test SequentialSampler op with various num_samples and num_repeats args combinations
    Expectation: Output is equal to the expected output
    """
    manifest_file = "../data/dataset/testManifestData/test5trainimgs.json"
    map_ = {(172876, 0): 0, (54214, 0): 1, (54214, 1): 2, (173673, 0): 3, (64631, 1): 4}

    def test_config(num_samples, num_repeats=None):
        sampler = ds.SequentialSampler(num_samples=num_samples)
        data1 = ds.ManifestDataset(manifest_file, sampler=sampler)
        if num_repeats is not None:
            data1 = data1.repeat(num_repeats)
        res = []
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            logger.info("item[image].shape[0]: {}, item[label].item(): {}"
                        .format(item["image"].shape[0], item["label"].item()))
            res.append(map_[(item["image"].shape[0], item["label"].item())])
        if print_res:
            logger.info("image.shapes and labels: {}".format(res))
        return res

    assert test_config(num_samples=3, num_repeats=None) == [0, 1, 2]
    assert test_config(num_samples=None, num_repeats=2) == [0, 1, 2, 3, 4] * 2
    assert test_config(num_samples=4, num_repeats=2) == [0, 1, 2, 3] * 2


def test_random_sampler(print_res=False):
    """
    Feature: RandomSampler op
    Description: Test RandomSampler with various replacement, num_samples, and num_repeats args combinations
    Expectation: Output is equal to the expected output
    """
    original_seed = config_get_set_seed(1234)
    manifest_file = "../data/dataset/testManifestData/test5trainimgs.json"
    map_ = {(172876, 0): 0, (54214, 0): 1, (54214, 1): 2, (173673, 0): 3, (64631, 1): 4}

    def test_config(replacement, num_samples, num_repeats):
        sampler = ds.RandomSampler(replacement=replacement, num_samples=num_samples)
        data1 = ds.ManifestDataset(manifest_file, sampler=sampler)
        data1 = data1.repeat(num_repeats)
        res = []
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(map_[(item["image"].shape[0], item["label"].item())])
        if print_res:
            logger.info("image.shapes and labels: {}".format(res))
        return res

    # this tests that each epoch COULD return different samples than the previous epoch
    assert len(set(test_config(replacement=False, num_samples=2, num_repeats=6))) > 2
    # the following two tests test replacement works
    ordered_res = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    assert sorted(test_config(replacement=False, num_samples=None, num_repeats=4)) == ordered_res
    assert sorted(test_config(replacement=True, num_samples=None, num_repeats=4)) != ordered_res

    ds.config.set_seed(original_seed)


def test_random_sampler_multi_iter(print_res=False):
    """
    Feature: RandomSampler op
    Description: Test RandomSampler with multiple iteration based on num_repeats
    Expectation: Output is equal to the expected output
    """
    manifest_file = "../data/dataset/testManifestData/test5trainimgs.json"
    map_ = {(172876, 0): 0, (54214, 0): 1, (54214, 1): 2, (173673, 0): 3, (64631, 1): 4}

    def test_config(replacement, num_samples, num_repeats, validate):
        sampler = ds.RandomSampler(replacement=replacement, num_samples=num_samples)
        data1 = ds.ManifestDataset(manifest_file, sampler=sampler)
        while num_repeats > 0:
            res = []
            for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
                res.append(map_[(item["image"].shape[0], item["label"].item())])
            if print_res:
                logger.info("image.shapes and labels: {}".format(res))
            if validate != sorted(res):
                break
            num_repeats -= 1
        assert num_repeats > 0

    test_config(replacement=True, num_samples=5, num_repeats=5, validate=[0, 1, 2, 3, 4, 5])


def test_sampler_py_api():
    """
    Feature: Sampler op
    Description: Test add_child op of a Sampler op to a Sampler op
    Expectation: Runs successfully
    """
    sampler = ds.SequentialSampler().parse()
    sampler1 = ds.RandomSampler().parse()
    sampler1.add_child(sampler)


def test_python_sampler():
    """
    Feature: Python Sampler op
    Description: Test Python Sampler op with and without inheritance
    Expectation: Output is equal to the expected output
    """
    manifest_file = "../data/dataset/testManifestData/test5trainimgs.json"
    map_ = {(172876, 0): 0, (54214, 0): 1, (54214, 1): 2, (173673, 0): 3, (64631, 1): 4}

    class Sp1(ds.Sampler):
        def __iter__(self):
            return iter([i for i in range(self.dataset_size)])

    class Sp2(ds.Sampler):
        def __init__(self, num_samples=None):
            super(Sp2, self).__init__(num_samples)
            # at this stage, self.dataset_size and self.num_samples are not yet known
            self.cnt = 0

        def __iter__(self):  # first epoch, all 0, second epoch all 1, third all 2 etc...
            return iter([self.cnt for i in range(self.num_samples)])

        def reset(self):
            self.cnt = (self.cnt + 1) % self.dataset_size

    def test_config(num_repeats, sampler):
        data1 = ds.ManifestDataset(manifest_file, sampler=sampler)
        if num_repeats is not None:
            data1 = data1.repeat(num_repeats)
        res = []
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            logger.info("item[image].shape[0]: {}, item[label].item(): {}"
                        .format(item["image"].shape[0], item["label"].item()))
            res.append(map_[(item["image"].shape[0], item["label"].item())])
        # print(res)
        return res

    def test_generator():
        class MySampler(ds.Sampler):
            def __iter__(self):
                for i in range(99, -1, -1):
                    yield i

        data1 = ds.GeneratorDataset([(np.array(i),) for i in range(100)], ["data"], sampler=MySampler())
        i = 99
        for data in data1:
            assert data[0].asnumpy() == (np.array(i),)
            i = i - 1

    # This 2nd case is the one that exhibits the same behavior as the case above without inheritance
    def test_generator_iter_sampler():
        class MySampler:
            def __iter__(self):
                for i in range(99, -1, -1):
                    yield i

        data1 = ds.GeneratorDataset([(np.array(i),) for i in range(100)], ["data"], sampler=MySampler())
        i = 99
        for data in data1:
            assert data[0].asnumpy() == (np.array(i),)
            i = i - 1

    assert test_config(2, Sp1(5)) == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    assert test_config(6, Sp2(2)) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0]
    test_generator()
    test_generator_iter_sampler()


def test_sequential_sampler2():
    """
    Feature: SequentialSampler op
    Description: Test SequentialSampler op with various start_index and num_samples args combinations
    Expectation: Output is equal to the expected output
    """
    manifest_file = "../data/dataset/testManifestData/test5trainimgs.json"
    map_ = {(172876, 0): 0, (54214, 0): 1, (54214, 1): 2, (173673, 0): 3, (64631, 1): 4}

    def test_config(start_index, num_samples):
        sampler = ds.SequentialSampler(start_index, num_samples)
        d = ds.ManifestDataset(manifest_file, sampler=sampler)

        res = []
        for item in d.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(map_[(item["image"].shape[0], item["label"].item())])

        return res

    assert test_config(0, 1) == [0]
    assert test_config(0, 2) == [0, 1]
    assert test_config(0, 3) == [0, 1, 2]
    assert test_config(0, 4) == [0, 1, 2, 3]
    assert test_config(0, 5) == [0, 1, 2, 3, 4]
    assert test_config(1, 1) == [1]
    assert test_config(2, 3) == [2, 3, 4]
    assert test_config(3, 2) == [3, 4]
    assert test_config(4, 1) == [4]
    assert test_config(4, None) == [4]


def test_subset_sampler():
    """
    Feature: SubsetSampler op
    Description: Test SubsetSampler op with various indices and num_samples args combinations including invalid ones
    Expectation: Output is equal to the expected output when input is valid, otherwise exception is raised
    """

    def test_config(indices, num_samples=None, exception_msg=None):
        def pipeline():
            sampler = ds.SubsetSampler(indices, num_samples)
            data = ds.NumpySlicesDataset(list(range(0, 10)), sampler=sampler)
            data2 = ds.NumpySlicesDataset(list(range(0, 10)), sampler=indices, num_samples=num_samples)
            dataset_size = data.get_dataset_size()
            dataset_size2 = data.get_dataset_size()
            res1 = [d[0] for d in data.create_tuple_iterator(num_epochs=1, output_numpy=True)], dataset_size
            res2 = [d[0] for d in data2.create_tuple_iterator(num_epochs=1, output_numpy=True)], dataset_size2
            return res1, res2

        if exception_msg is None:
            res, res2 = pipeline()
            res, size = res
            res2, size2 = res2
            if not isinstance(indices, list):
                indices = list(indices)
            assert indices[:num_samples] == res
            assert len(indices[:num_samples]) == size
            assert indices[:num_samples] == res2
            assert len(indices[:num_samples]) == size2
        else:
            with pytest.raises(Exception) as error_info:
                pipeline()
            print(str(error_info.value))
            assert exception_msg in str(error_info.value)

    test_config([1, 2, 3])
    test_config(list(range(10)))
    test_config([0])
    test_config([9])
    test_config(list(range(0, 10, 2)))
    test_config(list(range(1, 10, 2)))
    test_config(list(range(9, 0, -1)))
    test_config(list(range(9, 0, -2)))
    test_config(list(range(8, 0, -2)))
    test_config([0, 9, 3, 2])
    test_config([0, 0, 0, 0])
    test_config([0])
    test_config([0, 9, 3, 2], num_samples=2)
    test_config([0, 9, 3, 2], num_samples=5)

    test_config(np.array([1, 2, 3]))

    test_config([20], exception_msg="Sample ID (20) is out of bound, expected range [0, 9]")
    test_config([10], exception_msg="Sample ID (10) is out of bound, expected range [0, 9]")
    test_config([0, 9, 0, 500], exception_msg="Sample ID (500) is out of bound, expected range [0, 9]")
    test_config([0, 9, -6, 2], exception_msg="Sample ID (-6) is out of bound, expected range [0, 9]")
    # test_config([], exception_msg="Indices list is empty") # temporary until we check with MindDataset
    test_config([0, 9, 3, 2], num_samples=-1,
                exception_msg="num_samples exceeds the boundary between 0 and 9223372036854775807(INT64_MAX)")
    test_config(np.array([[1], [5]]), num_samples=10,
                exception_msg="SubsetSampler: Type of indices element must be int, but got list[0]: [1],"
                              " type: <class 'numpy.ndarray'>.")


def test_sampler_chain():
    """
    Feature: Chained Sampler
    Description: ManifestDataset with sampler chain; add SequentialSampler as a child for DistributedSampler
    Expectation: Correct error is raised as expected
    """
    manifest_file = "../data/dataset/testManifestData/test5trainimgs.json"
    map_ = {(172876, 0): 0, (54214, 0): 1, (54214, 1): 2, (173673, 0): 3, (64631, 1): 4}

    def test_config(num_shards, shard_id, start_index=0):
        sampler = ds.DistributedSampler(num_shards, shard_id, shuffle=False, num_samples=5)
        child_sampler = ds.SequentialSampler(start_index)
        sampler.add_child(child_sampler)

        data1 = ds.ManifestDataset(manifest_file, sampler=sampler)

        res = []
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            logger.info("item[image].shape[0]: {}, item[label].item(): {}"
                        .format(item["image"].shape[0], item["label"].item()))
            res.append(map_[(item["image"].shape[0], item["label"].item())])
        return res

    assert test_config(2, 0) == [0, 2, 4]
    assert test_config(2, 1) == [1, 3, 0]
    assert test_config(5, 0) == [0]
    assert test_config(5, 1) == [1]
    assert test_config(5, 2) == [2]
    assert test_config(5, 3) == [3]
    assert test_config(5, 4) == [4]
    assert test_config(2, 0, 1) == [1, 3]
    assert test_config(2, 1, 1) == [2, 4]
    assert test_config(5, 0, 1) == [1]
    assert test_config(5, 1, 1) == [2]
    assert test_config(5, 2, 1) == [3]
    assert test_config(5, 3, 1) == [4]
    assert test_config(5, 4, 1) == [1]


def test_add_sampler_invalid_input():
    """
    Feature: Sampler op
    Description: Test use_sampler op when the arg is not an instance of a sample and
        another separate case when num_samples and sampler are specified at the same time in dataset arg
    Expectation: Correct error is raised as expected
    """
    manifest_file = "../data/dataset/testManifestData/test5trainimgs.json"
    _ = {(172876, 0): 0, (54214, 0): 1, (54214, 1): 2, (173673, 0): 3, (64631, 1): 4}
    data1 = ds.ManifestDataset(manifest_file)

    with pytest.raises(TypeError) as info:
        data1.use_sampler(1)
    assert "not an instance of a sampler" in str(info.value)

    with pytest.raises(TypeError) as info:
        data1.use_sampler("sampler")
    assert "not an instance of a sampler" in str(info.value)

    sampler = ds.SequentialSampler()
    with pytest.raises(RuntimeError) as info:
        _ = ds.ManifestDataset(manifest_file, sampler=sampler, num_samples=20)
    assert "sampler and num_samples cannot be specified at the same time" in str(info.value)


def test_distributed_sampler_invalid_offset():
    """
    Feature: DistributedSampler op
    Description: Test DistributedSampler op when offset is more than num_shards
    Expectation: Error is raised as expected
    """
    with pytest.raises(RuntimeError) as info:
        _ = ds.DistributedSampler(num_shards=4, shard_id=0, shuffle=False, num_samples=None, offset=5).parse()
    assert "DistributedSampler: offset must be no more than num_shards(4)" in str(info.value)


def test_sampler_list():
    """
    Feature: Sampler op
    Description: Test various sampler args (int and not int) in ImageFolderDataset
    Expectation: Output is equal to the expected output when sampler has data type int, otherwise exception is raised
    """
    data1 = ds.ImageFolderDataset("../data/dataset/testPK/data", sampler=[1, 3, 5])
    data21 = ds.ImageFolderDataset("../data/dataset/testPK/data", shuffle=False).take(2).skip(1)
    data22 = ds.ImageFolderDataset("../data/dataset/testPK/data", shuffle=False).take(4).skip(3)
    data23 = ds.ImageFolderDataset("../data/dataset/testPK/data", shuffle=False).take(6).skip(5)

    dataset_equal(data1, data21 + data22 + data23, 0)

    data3 = ds.ImageFolderDataset("../data/dataset/testPK/data", sampler=1)
    dataset_equal(data3, data21, 0)

    def bad_pipeline(sampler, msg):
        with pytest.raises(Exception) as info:
            data1 = ds.ImageFolderDataset("../data/dataset/testPK/data", sampler=sampler)
            for _ in data1:
                pass
        assert msg in str(info.value)

    bad_pipeline(sampler=[1.5, 7],
                 msg="Type of indices element must be int, but got list[0]: 1.5, type: <class 'float'>")

    bad_pipeline(sampler=["a", "b"],
                 msg="Type of indices element must be int, but got list[0]: a, type: <class 'str'>.")
    bad_pipeline(sampler="a", msg="Unsupported sampler object of type (<class 'str'>)")
    bad_pipeline(sampler="", msg="Unsupported sampler object of type (<class 'str'>)")
    bad_pipeline(sampler=np.array([[1, 2]]),
                 msg="Type of indices element must be int, but got list[0]: [1 2], type: <class 'numpy.ndarray'>.")


def check_result(expected, result):
    for index, item in enumerate(result):
        assert str(expected[index][0]) == item


def test_sampler_when_less_and_larger_index_ids():
    """
    Feature: Sampler op
    Description: Test sampler with less and larger index ids
    Expectation: success
    """

    # sampler with less index ids
    class MySampler:
        def __iter__(self):
            for i in range(0, 10, 2):
                yield i

    np_data = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']

    dataset = ds.NumpySlicesDataset(np_data, column_names=["data"], sampler=MySampler())
    count = 0
    expected_data = []
    for data in dataset.create_tuple_iterator(num_epochs=1):
        count += 1
        expected_data.append(data)
    assert count == 5
    check_result(expected_data, ['a', 'c', 'e', 'g', 'i'])

    epochs = 3
    ds_iter = dataset.create_tuple_iterator(num_epochs=epochs)
    for _ in range(epochs):
        count = 0
        expected_data = []
        for data in ds_iter:
            count += 1
            expected_data.append(data)
        assert count == 5
        check_result(expected_data, ['a', 'c', 'e', 'g', 'i'])

    # sampler with larger index ids
    index = [3, 4, 3, 2, 0, 11, 5, 5, 5, 9, 1, 11, 11, 11, 11, 8]

    class MySampler2:
        def __iter__(self):
            for i in index:
                yield i

    dataset2 = ds.NumpySlicesDataset(np_data, column_names=["data"], sampler=MySampler2())
    count = 0
    expected_data = []
    for data in dataset2.create_tuple_iterator(num_epochs=1):
        count += 1
        expected_data.append(data)
    assert count == 16
    check_result(expected_data, ['d', 'e', 'd', 'c', 'a', 'l', 'f', 'f', 'f', 'j', 'b', 'l', 'l', 'l', 'l', 'i'])

    epochs = 3
    ds_iter2 = dataset2.create_tuple_iterator(num_epochs=epochs)
    for _ in range(epochs):
        count = 0
        expected_data = []
        for data in ds_iter2:
            count += 1
            expected_data.append(data)
        assert count == 16
        check_result(expected_data, ['d', 'e', 'd', 'c', 'a', 'l', 'f', 'f', 'f', 'j', 'b', 'l', 'l', 'l', 'l', 'i'])


def test_sampler_with_getitem_method():
    """
    Feature: Sampler op
    Description: Test sampler with __getitem__ method
    Expectation: success
    """

    # sampler with equal index ids
    class MySampler:
        def __init__(self):
            self.index_ids = [3, 8, 7, 2, 0, 9, 11, 4, 5, 1, 6, 10]

        def __getitem__(self, index):
            return self.index_ids[index]

        def __len__(self):
            return len(self.index_ids)

    np_data = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']

    dataset = ds.NumpySlicesDataset(np_data, column_names=["data"], sampler=MySampler())
    count = 0
    expected_data = []
    for data in dataset.create_tuple_iterator(num_epochs=1):
        count += 1
        expected_data.append(data)
    assert count == 12
    check_result(expected_data, ['d', 'i', 'h', 'c', 'a', 'j', 'l', 'e', 'f', 'b', 'g', 'k'])

    epochs = 3
    ds_iter = dataset.create_tuple_iterator(num_epochs=epochs)
    for _ in range(epochs):
        count = 0
        expected_data = []
        for data in ds_iter:
            count += 1
            expected_data.append(data)
        assert count == 12
        check_result(expected_data, ['d', 'i', 'h', 'c', 'a', 'j', 'l', 'e', 'f', 'b', 'g', 'k'])

    # sampler with less index ids
    class MySampler2:
        def __init__(self):
            self.index_ids = [0, 2, 4, 6, 8]

        def __getitem__(self, index):
            return self.index_ids[index]

        def __len__(self):
            return len(self.index_ids)

    np_data = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']

    dataset2 = ds.NumpySlicesDataset(np_data, column_names=["data"], sampler=MySampler2())
    count = 0
    expected_data = []
    for data in dataset2.create_tuple_iterator(num_epochs=1):
        count += 1
        expected_data.append(data)
    assert count == 5
    check_result(expected_data, ['a', 'c', 'e', 'g', 'i'])

    epochs = 3
    ds_iter2 = dataset2.create_tuple_iterator(num_epochs=epochs)
    for _ in range(epochs):
        count = 0
        expected_data = []
        for data in ds_iter2:
            count += 1
            expected_data.append(data)
        assert count == 5
        check_result(expected_data, ['a', 'c', 'e', 'g', 'i'])

    # sampler with larger index ids
    class MySampler3:
        def __init__(self):
            self.index_ids = [3, 4, 3, 2, 0, 11, 5, 5, 5, 9, 1, 11, 11, 11, 11, 8]

        def __getitem__(self, index):
            return self.index_ids[index]

        def __len__(self):
            return len(self.index_ids)

    dataset3 = ds.NumpySlicesDataset(np_data, column_names=["data"], sampler=MySampler3())
    count = 0
    expected_data = []
    for data in dataset3.create_tuple_iterator(num_epochs=1):
        count += 1
        expected_data.append(data)
    assert count == 16
    check_result(expected_data, ['d', 'e', 'd', 'c', 'a', 'l', 'f', 'f', 'f', 'j', 'b', 'l', 'l', 'l', 'l', 'i'])

    epochs = 3
    ds_iter3 = dataset3.create_tuple_iterator(num_epochs=epochs)
    for _ in range(epochs):
        count = 0
        expected_data = []
        for data in ds_iter3:
            count += 1
            expected_data.append(data)
        assert count == 16
        check_result(expected_data, ['d', 'e', 'd', 'c', 'a', 'l', 'f', 'f', 'f', 'j', 'b', 'l', 'l', 'l', 'l', 'i'])


@pytest.mark.parametrize("num_shards", (1, 2, 3))
def test_distributed_sampler_offset(num_shards):
    """
    Feature: DistributedSampler
    Description: Test DistributedSampler with offset
    Expectation: Result is as expected
    """
    dataset_size = 10000
    for shard_id in range(num_shards):
        for offset in range(-1, num_shards):
            distributed_sampler = ds.DistributedSampler(num_shards=num_shards, shard_id=shard_id, offset=offset)
            dataset = ds.MnistDataset("../data/dataset/testMnistData", sampler=distributed_sampler)
            if offset == -1 or offset == shard_id:
                expected_result = (dataset_size -1) // num_shards + 1
            else:
                expected_result = dataset_size // num_shards
            assert dataset.get_dataset_size() == expected_result
            assert sum([1 for _ in dataset]) == expected_result


class StrNumberDataset:
    def __init__(self):
        self.data = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def test_python_sampler_randomness():
    """
    Feature: PythonSampler
    Description: Test PythonSampler randomness consistency
    Expectation: Result is as expected
    """

    class UDFSampler(ds.Sampler):
        def __init__(self):
            super().__init__()
            self.generator = np.random.default_rng(seed=0)

        def __iter__(self):
            # number of samples is random
            num_samples = self.generator.integers(1, 11)
            indices = [i for i in range(num_samples)]
            # order of samples is random
            self.generator.shuffle(indices)
            return iter(indices)

    dataset = ds.GeneratorDataset(StrNumberDataset(), column_names=["data"], sampler=UDFSampler())
    assert dataset.get_dataset_size() == 9

    num_epochs = 3

    # iterate the sampler to get the data manually
    fake_sampler = UDFSampler()
    fake_dataset = StrNumberDataset()
    expected_data = []
    for _ in range(num_epochs):
        for index in fake_sampler:
            data = fake_dataset[index]
            expected_data.append(np.array(data))

    # iterate the dataset 2 times to get the data
    total_result = []
    for _ in range(2):
        result = []
        dataset_iter = dataset.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)
        for _ in range(num_epochs):
            for data in dataset_iter:
                result.append(data["data"])
        total_result.append(result)

    assert len(total_result[0]) == len(expected_data)
    assert len(total_result[1]) == len(expected_data)
    for data_first_time, data_second_time, data_expect in zip(total_result[0], total_result[1], expected_data):
        # every iteration should have the same data as expected
        np.testing.assert_array_equal(data_first_time, data_expect)
        np.testing.assert_array_equal(data_second_time, data_expect)


@pytest.mark.parametrize("num_samples", (None, 5, 10, 15))
def test_python_sampler_with_num_samples(num_samples):
    """
    Feature: PythonSampler
    Description: Test Python sampler with different num_samples
    Expectation: Result is as expected
    """

    class UDFSampler(ds.Sampler):
        def __iter__(self):
            yield from range(10)

    sampler = UDFSampler(num_samples)  # specify number of samples
    dataset = ds.GeneratorDataset(StrNumberDataset(), column_names=["data"], sampler=sampler)
    expected_num_samples = num_samples if num_samples is not None else 10
    assert dataset.get_dataset_size() == expected_num_samples
    if expected_num_samples <= 10:
        expected_result = np.array([str(i) for i in range(expected_num_samples)])
        for index, item in enumerate(dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
            np.testing.assert_array_equal(item["data"], expected_result[index])
    else:
        # num_samples is larger than the actual sampler length
        with pytest.raises(RuntimeError) as error:
            for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
                pass
        assert "The actual amount of data read from generator 10 is different from generator.len 15" in str(error.value)


@pytest.mark.parametrize("inherit_sampler", (True, False))
@pytest.mark.parametrize("length", (5, 10, 15))
def test_python_sampler_with_length(inherit_sampler, length):
    """
    Feature: PythonSampler
    Description: Test PythonSampler with __len__ method
    Expectation: Result is as expected
    """

    if inherit_sampler:
        # User defined sampler inherits from ds.Sampler
        class UDFSampler(ds.Sampler):
            def __iter__(self):
                yield from range(10)

            def __len__(self):
                return length
    else:
        # User defined sampler does not inherit from ds.Sampler
        class UDFSampler:
            def __iter__(self):
                yield from range(10)

            def __len__(self):
                return length

    sampler = UDFSampler()
    dataset = ds.GeneratorDataset(StrNumberDataset(), column_names=["data"], sampler=sampler)
    assert dataset.get_dataset_size() == length

    if length <= 10:
        expected_result = np.array([str(i) for i in range(length)])
        for index, item in enumerate(dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
            np.testing.assert_array_equal(item["data"], expected_result[index])
    else:
        # length from __len__ is larger than the actual sampler length
        with pytest.raises(RuntimeError) as error:
            for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
                pass
        assert "The actual amount of data read from generator 10 is different from generator.len 15" in str(error.value)


if __name__ == '__main__':
    test_sequential_sampler(True)
    test_random_sampler(True)
    test_random_sampler_multi_iter(True)
    test_sampler_py_api()
    test_python_sampler()
    test_sequential_sampler2()
    test_subset_sampler()
    test_sampler_chain()
    test_add_sampler_invalid_input()
    test_distributed_sampler_invalid_offset()
    test_sampler_list()
    test_sampler_when_less_and_larger_index_ids()
    test_sampler_with_getitem_method()
    test_distributed_sampler_offset(num_shards=1)
    test_python_sampler_randomness()
    test_python_sampler_with_num_samples(num_samples=None)
    test_python_sampler_with_length(inherit_sampler=True, length=10)
