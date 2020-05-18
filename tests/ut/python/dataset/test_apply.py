# Copyright 2019 Huawei Technologies Co., Ltd
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
import mindspore.dataset as ds
from mindspore import log as logger
import mindspore.dataset.transforms.vision.c_transforms as vision
import numpy as np

DATA_DIR = "../data/dataset/testPK/data"


# Generate 1d int numpy array from 0 - 64
def generator_1d():
    for i in range(64):
        yield (np.array([i]),)


def test_apply_generator_case():
    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    data2 = ds.GeneratorDataset(generator_1d, ["data"])

    def dataset_fn(ds):
        ds = ds.repeat(2)
        return ds.batch(4)

    data1 = data1.apply(dataset_fn)
    data2 = data2.repeat(2)
    data2 = data2.batch(4)

    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        assert np.array_equal(item1["data"], item2["data"])


def test_apply_imagefolder_case():
    # apply dataset map operations
    data1 = ds.ImageFolderDatasetV2(DATA_DIR, num_shards=4, shard_id=3)
    data2 = ds.ImageFolderDatasetV2(DATA_DIR, num_shards=4, shard_id=3)

    decode_op = vision.Decode()
    normalize_op = vision.Normalize([121.0, 115.0, 100.0], [70.0, 68.0, 71.0])

    def dataset_fn(ds):
        ds = ds.map(operations=decode_op)
        ds = ds.map(operations=normalize_op)
        ds = ds.repeat(2)
        return ds

    data1 = data1.apply(dataset_fn)
    data2 = data2.map(operations=decode_op)
    data2 = data2.map(operations=normalize_op)
    data2 = data2.repeat(2)

    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        assert np.array_equal(item1["image"], item2["image"])


def test_apply_flow_case_0(id=0):
    # apply control flow operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    def dataset_fn(ds):
        if id == 0:
            ds = ds.batch(4)
        elif id == 1:
            ds = ds.repeat(2)
        elif id == 2:
            ds = ds.batch(4)
            ds = ds.repeat(2)
        else:
            ds = ds.shuffle(buffer_size=4)
        return ds

    data1 = data1.apply(dataset_fn)
    num_iter = 0
    for _ in data1.create_dict_iterator():
        num_iter = num_iter + 1

    if id == 0:
        assert num_iter == 16
    elif id == 1:
        assert num_iter == 128
    elif id == 2:
        assert num_iter == 32
    else:
        assert num_iter == 64


def test_apply_flow_case_1(id=1):
    # apply control flow operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    def dataset_fn(ds):
        if id == 0:
            ds = ds.batch(4)
        elif id == 1:
            ds = ds.repeat(2)
        elif id == 2:
            ds = ds.batch(4)
            ds = ds.repeat(2)
        else:
            ds = ds.shuffle(buffer_size=4)
        return ds

    data1 = data1.apply(dataset_fn)
    num_iter = 0
    for _ in data1.create_dict_iterator():
        num_iter = num_iter + 1

    if id == 0:
        assert num_iter == 16
    elif id == 1:
        assert num_iter == 128
    elif id == 2:
        assert num_iter == 32
    else:
        assert num_iter == 64


def test_apply_flow_case_2(id=2):
    # apply control flow operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    def dataset_fn(ds):
        if id == 0:
            ds = ds.batch(4)
        elif id == 1:
            ds = ds.repeat(2)
        elif id == 2:
            ds = ds.batch(4)
            ds = ds.repeat(2)
        else:
            ds = ds.shuffle(buffer_size=4)
        return ds

    data1 = data1.apply(dataset_fn)
    num_iter = 0
    for _ in data1.create_dict_iterator():
        num_iter = num_iter + 1

    if id == 0:
        assert num_iter == 16
    elif id == 1:
        assert num_iter == 128
    elif id == 2:
        assert num_iter == 32
    else:
        assert num_iter == 64


def test_apply_flow_case_3(id=3):
    # apply control flow operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    def dataset_fn(ds):
        if id == 0:
            ds = ds.batch(4)
        elif id == 1:
            ds = ds.repeat(2)
        elif id == 2:
            ds = ds.batch(4)
            ds = ds.repeat(2)
        else:
            ds = ds.shuffle(buffer_size=4)
        return ds

    data1 = data1.apply(dataset_fn)
    num_iter = 0
    for _ in data1.create_dict_iterator():
        num_iter = num_iter + 1

    if id == 0:
        assert num_iter == 16
    elif id == 1:
        assert num_iter == 128
    elif id == 2:
        assert num_iter == 32
    else:
        assert num_iter == 64


def test_apply_exception_case():
    # apply exception operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    def dataset_fn(ds):
        ds = ds.repeat(2)
        return ds.batch(4)

    def exception_fn(ds):
        return np.array([[0], [1], [3], [4], [5]])

    try:
        data1 = data1.apply("123")
        for _ in data1.create_dict_iterator():
            pass
        assert False
    except TypeError:
        pass

    try:
        data1 = data1.apply(exception_fn)
        for _ in data1.create_dict_iterator():
            pass
        assert False
    except TypeError:
        pass

    try:
        data2 = data1.apply(dataset_fn)
        data3 = data1.apply(dataset_fn)
        for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
            pass
        assert False
    except ValueError:
        pass


if __name__ == '__main__':
    logger.info("Running test_apply.py test_apply_generator_case() function")
    test_apply_generator_case()

    logger.info("Running test_apply.py test_apply_imagefolder_case() function")
    test_apply_imagefolder_case()

    logger.info("Running test_apply.py test_apply_flow_case(id) function")
    test_apply_flow_case_0()
    test_apply_flow_case_1()
    test_apply_flow_case_2()
    test_apply_flow_case_3()

    logger.info("Running test_apply.py test_apply_exception_case() function")
    test_apply_exception_case()
