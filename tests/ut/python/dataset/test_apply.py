# Copyright 2019-2022 Huawei Technologies Co., Ltd
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

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger

DATA_DIR = "../data/dataset/testPK/data"


# Generate 1d int numpy array from 0 - 64
def generator_1d():
    for i in range(64):
        yield (np.array([i]),)


def test_apply_generator_case():
    """
    Feature: Apply op
    Description: Test apply op on GeneratorDataset
    Expectation: Output is equal to the expected output
    """
    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    data2 = ds.GeneratorDataset(generator_1d, ["data"])

    def dataset_fn(ds_):
        ds_ = ds_.repeat(2)
        return ds_.batch(4)

    data1 = data1.apply(dataset_fn)
    data2 = data2.repeat(2)
    data2 = data2.batch(4)

    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1["data"], item2["data"])


def test_apply_imagefolder_case():
    """
    Feature: Apply op
    Description: Test apply op on ImageFolderDataset
    Expectation: Output is equal to the expected output
    """
    # apply dataset map operations
    data1 = ds.ImageFolderDataset(DATA_DIR, num_shards=4, shard_id=3)
    data2 = ds.ImageFolderDataset(DATA_DIR, num_shards=4, shard_id=3)

    decode_op = vision.Decode()
    normalize_op = vision.Normalize([121.0, 115.0, 100.0], [70.0, 68.0, 71.0], True)

    def dataset_fn(ds_):
        ds_ = ds_.map(operations=decode_op)
        ds_ = ds_.map(operations=normalize_op)
        ds_ = ds_.repeat(2)
        return ds_

    data1 = data1.apply(dataset_fn)
    data2 = data2.map(operations=decode_op)
    data2 = data2.map(operations=normalize_op)
    data2 = data2.repeat(2)

    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1["image"], item2["image"])


def test_apply_flow_case_0(id_=0):
    """
    Feature: Apply op
    Description: Test control flow operation by applying batch op
    Expectation: Output is equal to the expected output
    """
    # apply control flow operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    def dataset_fn(ds_):
        if id_ == 0:
            ds_ = ds_.batch(4)
        elif id_ == 1:
            ds_ = ds_.repeat(2)
        elif id_ == 2:
            ds_ = ds_.batch(4)
            ds_ = ds_.repeat(2)
        else:
            ds_ = ds_.shuffle(buffer_size=4)
        return ds_

    data1 = data1.apply(dataset_fn)
    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    if id_ == 0:
        assert num_iter == 16
    elif id_ == 1:
        assert num_iter == 128
    elif id_ == 2:
        assert num_iter == 32
    else:
        assert num_iter == 64


def test_apply_flow_case_1(id_=1):
    """
    Feature: Apply op
    Description: Test control flow operation by applying repeat op
    Expectation: Output is equal to the expected output
    """
    # apply control flow operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    def dataset_fn(ds_):
        if id_ == 0:
            ds_ = ds_.batch(4)
        elif id_ == 1:
            ds_ = ds_.repeat(2)
        elif id_ == 2:
            ds_ = ds_.batch(4)
            ds_ = ds_.repeat(2)
        else:
            ds_ = ds_.shuffle(buffer_size=4)
        return ds_

    data1 = data1.apply(dataset_fn)
    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    if id_ == 0:
        assert num_iter == 16
    elif id_ == 1:
        assert num_iter == 128
    elif id_ == 2:
        assert num_iter == 32
    else:
        assert num_iter == 64


def test_apply_flow_case_2(id_=2):
    """
    Feature: Apply op
    Description: Test control flow operation by applying batch op then repeat op
    Expectation: Output is equal to the expected output
    """
    # apply control flow operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    def dataset_fn(ds_):
        if id_ == 0:
            ds_ = ds_.batch(4)
        elif id_ == 1:
            ds_ = ds_.repeat(2)
        elif id_ == 2:
            ds_ = ds_.batch(4)
            ds_ = ds_.repeat(2)
        else:
            ds_ = ds_.shuffle(buffer_size=4)
        return ds_

    data1 = data1.apply(dataset_fn)
    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    if id_ == 0:
        assert num_iter == 16
    elif id_ == 1:
        assert num_iter == 128
    elif id_ == 2:
        assert num_iter == 32
    else:
        assert num_iter == 64


def test_apply_flow_case_3(id_=3):
    """
    Feature: Apply op
    Description: Test control flow operation by applying shuffle op
    Expectation: Output is equal to the expected output
    """
    # apply control flow operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    def dataset_fn(ds_):
        if id_ == 0:
            ds_ = ds_.batch(4)
        elif id_ == 1:
            ds_ = ds_.repeat(2)
        elif id_ == 2:
            ds_ = ds_.batch(4)
            ds_ = ds_.repeat(2)
        else:
            ds_ = ds_.shuffle(buffer_size=4)
        return ds_

    data1 = data1.apply(dataset_fn)
    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    if id_ == 0:
        assert num_iter == 16
    elif id_ == 1:
        assert num_iter == 128
    elif id_ == 2:
        assert num_iter == 32
    else:
        assert num_iter == 64


def test_apply_exception_case():
    """
    Feature: Apply op
    Description: Test apply op using invalid inputs
    Expectation: Correct error is raised as expected
    """
    # apply exception operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    def dataset_fn(ds_):
        ds_ = ds_.repeat(2)
        return ds_.batch(4)

    def exception_fn():
        return np.array([[0], [1], [3], [4], [5]])

    try:
        data1 = data1.apply("123")
        for _ in data1.create_dict_iterator(num_epochs=1):
            pass
        assert False
    except TypeError:
        pass

    try:
        data1 = data1.apply(exception_fn)
        for _ in data1.create_dict_iterator(num_epochs=1):
            pass
        assert False
    except TypeError:
        pass

    try:
        data2 = data1.apply(dataset_fn)
        _ = data1.apply(dataset_fn)
        for _, _ in zip(data1.create_dict_iterator(num_epochs=1), data2.create_dict_iterator(num_epochs=1)):
            pass
        assert False
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))


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
