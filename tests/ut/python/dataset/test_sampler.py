# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from util import dataset_equal


# test5trainimgs.json contains 5 images whose un-decoded shape is [83554, 54214, 65512, 54214, 64631]
# the label of each image is [0,0,0,1,1] each image can be uniquely identified
# via the following lookup table (dict){(83554, 0): 0, (54214, 0): 1, (54214, 1): 2, (65512, 0): 3, (64631, 1): 4}

def test_sequential_sampler(print_res=False):
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


def test_random_sampler_multi_iter(print_res=False):
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
    sampler = ds.SequentialSampler().parse()
    sampler1 = ds.RandomSampler().parse()
    sampler1.add_child(sampler)


def test_python_sampler():
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

        def __iter__(self):  # first epoch, all 0, second epoch all 1, third all 2 etc.. ...
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
        class MySampler():
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
    manifest_file = "../data/dataset/testManifestData/test5trainimgs.json"
    map_ = {(172876, 0): 0, (54214, 0): 1, (54214, 1): 2, (173673, 0): 3, (64631, 1): 4}

    def test_config(num_shards, shard_id):
        sampler = ds.DistributedSampler(num_shards, shard_id, shuffle=False, num_samples=5)
        child_sampler = ds.SequentialSampler()
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


def test_add_sampler_invalid_input():
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
        data2 = ds.ManifestDataset(manifest_file, sampler=sampler, num_samples=20)
    assert "sampler and num_samples cannot be specified at the same time" in str(info.value)


def test_distributed_sampler_invalid_offset():
    with pytest.raises(RuntimeError) as info:
        sampler = ds.DistributedSampler(num_shards=4, shard_id=0, shuffle=False, num_samples=None, offset=5).parse()
    assert "DistributedSampler: offset must be no more than num_shards(4)" in str(info.value)


def test_sampler_list():
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
