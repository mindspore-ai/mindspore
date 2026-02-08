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
"""
This is the test module for mindrecord
"""
import os
import pytest
import numpy as np

import mindspore.dataset as ds
from mindspore import log as logger
from mindspore.dataset import Shuffle
from mindspore.mindrecord import FileWriter
from mindspore.dataset import vision
import mindspore.common.dtype as mstype
from mindspore.dataset import transforms
from mindspore.dataset.vision import Border
from mindspore.dataset.vision import Inter
from mindspore.dataset.callback import DSCallback
import mindspore.dataset.vision.utils as mode
from util import config_get_set_seed

FILES_NUM = 4
CV_DIR_NAME = "../data/mindrecord/testImageNetData"
DATA_FILE = "../data/dataset/testTextFileDataset/1.txt"
MINDRECORD_IMAGENET = "../data/mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0"
image_jpg = "../data/dataset/apple.jpg"

@pytest.fixture
def add_and_remove_cv_file():
    """add/remove cv file"""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    paths = ["{}{}".format(file_name, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    try:
        for x in paths:
            if os.path.exists("{}".format(x)):
                os.remove("{}".format(x))
            if os.path.exists("{}.db".format(x)):
                os.remove("{}.db".format(x))
        writer = FileWriter(file_name, FILES_NUM)
        data = get_data(CV_DIR_NAME, True)
        cv_schema_json = {"id": {"type": "int32"},
                          "file_name": {"type": "string"},
                          "label": {"type": "int32"},
                          "data": {"type": "bytes"}}
        writer.add_schema(cv_schema_json, "img_schema")
        writer.add_index(["file_name", "label"])
        writer.write_raw_data(data)
        writer.commit()
        yield "yield_cv_data"
    except Exception as error:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))
        raise error

    for x in paths:
        os.remove("{}".format(x))
        os.remove("{}.db".format(x))


def test_cv_minddataset_pk_sample_no_column(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset with PKSampler without any columns_list in the dataset
    Expectation: Output is equal to the expected output
    """
    num_readers = 4
    sampler = ds.PKSampler(2)
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", None, num_readers,
                              sampler=sampler)

    assert data_set.get_dataset_size() == 6
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[file_name]: \
                {}------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1


def test_cv_minddataset_pk_sample_basic(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test basic read MindDataset with PKSampler
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.PKSampler(2)
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              sampler=sampler)

    assert data_set.get_dataset_size() == 6
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[data]: \
                {}------------------------".format(item["data"][:10]))
        logger.info("-------------- item[file_name]: \
                {}------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1


def test_cv_minddataset_pk_sample_shuffle(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset with PKSampler with shuffle=True
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.PKSampler(3, None, True)
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              sampler=sampler)

    assert data_set.get_dataset_size() == 9
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[file_name]: \
                {}------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 9


def test_cv_minddataset_pk_sample_shuffle_1(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset with PKSampler with shuffle=True and
        with num_samples larger than get_dataset_size
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.PKSampler(3, None, True, 'label', 5)
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              sampler=sampler)

    assert data_set.get_dataset_size() == 5
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[file_name]: \
                {}------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 5


def test_cv_minddataset_pk_sample_shuffle_2(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset with PKSampler with shuffle=True and
        with num_samples larger than get_dataset_size
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.PKSampler(3, None, True, 'label', 10)
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              sampler=sampler)

    assert data_set.get_dataset_size() == 9
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[file_name]: \
                {}------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 9


def test_cv_minddataset_pk_sample_out_of_range_0(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset with PKSampler with shuffle=True and num_val that is out of range
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.PKSampler(5, None, True)
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              sampler=sampler)
    assert data_set.get_dataset_size() == 15
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[file_name]: \
                {}------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 15


def test_cv_minddataset_pk_sample_out_of_range_1(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset with PKSampler with shuffle=True, num_val that is out of range, and
        num_samples larger than get_dataset_size
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.PKSampler(5, None, True, 'label', 20)
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              sampler=sampler)
    assert data_set.get_dataset_size() == 15
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[file_name]: \
                {}------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 15


def test_cv_minddataset_pk_sample_out_of_range_2(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset with PKSampler with shuffle=True, num_val that is out of range, and
        num_samples that is equal to get_dataset_size
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.PKSampler(5, None, True, 'label', 10)
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              sampler=sampler)
    assert data_set.get_dataset_size() == 10
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[file_name]: \
                {}------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 10


def test_cv_minddataset_subset_random_sample_basic(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test basic read MindDataset with SubsetRandomSampler
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    indices = [1, 2, 3, 5, 7]
    samplers = (ds.SubsetRandomSampler(indices), ds.SubsetSampler(indices))
    for sampler in samplers:
        data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                                  sampler=sampler)
        assert data_set.get_dataset_size() == 5
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            logger.info(
                "-------------- cv reader basic: {} ------------------------".format(num_iter))
            logger.info(
                "-------------- item[data]: {}  -----------------------------".format(item["data"]))
            logger.info(
                "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
            logger.info(
                "-------------- item[label]: {} ----------------------------".format(item["label"]))
            num_iter += 1
        assert num_iter == 5


def test_cv_minddataset_subset_random_sample_replica(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset with SubsetRandomSampler with duplicate index in the indices
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    indices = [1, 2, 2, 5, 7, 9]
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    samplers = ds.SubsetRandomSampler(indices), ds.SubsetSampler(indices)
    for sampler in samplers:
        data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                                  sampler=sampler)
        assert data_set.get_dataset_size() == 6
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            logger.info(
                "-------------- cv reader basic: {} ------------------------".format(num_iter))
            logger.info(
                "-------------- item[data]: {}  -----------------------------".format(item["data"]))
            logger.info(
                "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
            logger.info(
                "-------------- item[label]: {} ----------------------------".format(item["label"]))
            num_iter += 1
        assert num_iter == 6


def test_cv_minddataset_subset_random_sample_empty(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset with SubsetRandomSampler with empty indices
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    indices = []
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    samplers = ds.SubsetRandomSampler(indices), ds.SubsetSampler(indices)
    for sampler in samplers:
        data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                                  sampler=sampler)
        assert data_set.get_dataset_size() == 0
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            logger.info(
                "-------------- cv reader basic: {} ------------------------".format(num_iter))
            logger.info(
                "-------------- item[data]: {}  -----------------------------".format(item["data"]))
            logger.info(
                "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
            logger.info(
                "-------------- item[label]: {} ----------------------------".format(item["label"]))
            num_iter += 1
        assert num_iter == 0


def test_cv_minddataset_subset_random_sample_out_of_range(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset with SubsetRandomSampler with indices that are out of range
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    indices = [1, 2, 4, 11, 13]
    samplers = ds.SubsetRandomSampler(indices), ds.SubsetSampler(indices)
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    for sampler in samplers:
        data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                                  sampler=sampler)
        assert data_set.get_dataset_size() == 5
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            logger.info(
                "-------------- cv reader basic: {} ------------------------".format(num_iter))
            logger.info(
                "-------------- item[data]: {}  -----------------------------".format(item["data"]))
            logger.info(
                "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
            logger.info(
                "-------------- item[label]: {} ----------------------------".format(item["label"]))
            num_iter += 1
        assert num_iter == 5


def test_cv_minddataset_subset_random_sample_negative(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset with SubsetRandomSampler with negative indices
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    indices = [1, 2, 4, -1, -2]
    samplers = ds.SubsetRandomSampler(indices), ds.SubsetSampler(indices)
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    for sampler in samplers:
        data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                                  sampler=sampler)
        assert data_set.get_dataset_size() == 5
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            logger.info(
                "-------------- cv reader basic: {} ------------------------".format(num_iter))
            logger.info(
                "-------------- item[data]: {}  -----------------------------".format(item["data"]))
            logger.info(
                "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
            logger.info(
                "-------------- item[label]: {} ----------------------------".format(item["label"]))
            num_iter += 1
        assert num_iter == 5


def test_cv_minddataset_random_sampler_basic(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test basic read MindDataset with RandomSampler
    Expectation: Output is equal to the expected output
    """
    data = get_data(CV_DIR_NAME, True)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.RandomSampler()
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              sampler=sampler)
    assert data_set.get_dataset_size() == 10
    num_iter = 0
    new_dataset = []
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
        new_dataset.append(item['file_name'])
    assert num_iter == 10
    assert new_dataset != [x['file_name'] for x in data]


def test_cv_minddataset_random_sampler_repeat(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset with RandomSampler followed by Repeat op
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    sampler = ds.RandomSampler()
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              sampler=sampler)
    assert data_set.get_dataset_size() == 10
    ds1 = data_set.repeat(3)
    num_iter = 0
    epoch1_dataset = []
    epoch2_dataset = []
    epoch3_dataset = []
    for item in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
        if num_iter <= 10:
            epoch1_dataset.append(item['file_name'])
        elif num_iter <= 20:
            epoch2_dataset.append(item['file_name'])
        else:
            epoch3_dataset.append(item['file_name'])
    assert num_iter == 30
    assert epoch1_dataset not in (epoch2_dataset, epoch3_dataset)
    assert epoch2_dataset not in (epoch1_dataset, epoch3_dataset)
    assert epoch3_dataset not in (epoch1_dataset, epoch2_dataset)


def test_cv_minddataset_random_sampler_replacement(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset with RandomSampler with replacement=True
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    sampler = ds.RandomSampler(replacement=True, num_samples=5)
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              sampler=sampler)
    assert data_set.get_dataset_size() == 5
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 5


def test_cv_minddataset_random_sampler_replacement_false_1(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset with RandomSampler with replacement=False and num_samples <= dataset size
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    sampler = ds.RandomSampler(replacement=False, num_samples=2)
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              sampler=sampler)
    assert data_set.get_dataset_size() == 2
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 2


def test_cv_minddataset_random_sampler_replacement_false_2(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset with RandomSampler with replacement=False and num_samples > dataset size
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    sampler = ds.RandomSampler(replacement=False, num_samples=20)
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              sampler=sampler)
    assert data_set.get_dataset_size() == 10
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 10


def test_cv_minddataset_sequential_sampler_basic(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test basic read MindDataset with SequentialSampler
    Expectation: Output is equal to the expected output
    """
    data = get_data(CV_DIR_NAME, True)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    sampler = ds.SequentialSampler(1, 4)
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              sampler=sampler)
    assert data_set.get_dataset_size() == 4
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        assert item['file_name'] == np.array(data[num_iter + 1]['file_name'])
        num_iter += 1
    assert num_iter == 4


def test_cv_minddataset_sequential_sampler_offeset(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset with SequentialSampler with offset on starting index
    Expectation: Output is equal to the expected output
    """
    data = get_data(CV_DIR_NAME, True)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    sampler = ds.SequentialSampler(2, 10)
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              sampler=sampler)
    dataset_size = data_set.get_dataset_size()
    assert dataset_size == 8
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        assert item['file_name'] == np.array(data[(num_iter + 2) % 10]['file_name'])
        num_iter += 1
    assert num_iter == 8


def test_cv_minddataset_sequential_sampler_exceed_size(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset with SequentialSampler with offset on starting index and
        num_samples > dataset size
    Expectation: Output is equal to the expected output
    """
    data = get_data(CV_DIR_NAME, True)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    sampler = ds.SequentialSampler(2, 20)
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              sampler=sampler)
    dataset_size = data_set.get_dataset_size()
    assert dataset_size == 8
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        assert item['file_name'] == np.array(data[(num_iter + 2) % 10]['file_name'])
        num_iter += 1
    assert num_iter == 8


def test_cv_minddataset_split_basic(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test basic read MindDataset after Split op is applied
    Expectation: Output is equal to the expected output
    """
    data = get_data(CV_DIR_NAME, True)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    d = ds.MindDataset(file_name + "0", columns_list,
                       num_readers, shuffle=False)
    d1, d2 = d.split([8, 2], randomize=False)
    assert d.get_dataset_size() == 10
    assert d1.get_dataset_size() == 8
    assert d2.get_dataset_size() == 2
    num_iter = 0
    for item in d1.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        assert item['file_name'] == np.array(data[num_iter]['file_name'])
        num_iter += 1
    assert num_iter == 8
    num_iter = 0
    for item in d2.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        assert item['file_name'] == np.array(data[num_iter + 8]['file_name'])
        num_iter += 1
    assert num_iter == 2


def test_cv_minddataset_split_exact_percent(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset after Split op is applied using exact percentages
    Expectation: Output is equal to the expected output
    """
    data = get_data(CV_DIR_NAME, True)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    d = ds.MindDataset(file_name + "0", columns_list,
                       num_readers, shuffle=False)
    d1, d2 = d.split([0.8, 0.2], randomize=False)
    assert d.get_dataset_size() == 10
    assert d1.get_dataset_size() == 8
    assert d2.get_dataset_size() == 2
    num_iter = 0
    for item in d1.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        assert item['file_name'] == np.array(data[num_iter]['file_name'])
        num_iter += 1
    assert num_iter == 8
    num_iter = 0
    for item in d2.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        assert item['file_name'] == np.array(data[num_iter + 8]['file_name'])
        num_iter += 1
    assert num_iter == 2


def test_cv_minddataset_split_fuzzy_percent(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset after Split op is applied using fuzzy percentages
    Expectation: Output is equal to the expected output
    """
    data = get_data(CV_DIR_NAME, True)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    d = ds.MindDataset(file_name + "0", columns_list,
                       num_readers, shuffle=False)
    d1, d2 = d.split([0.41, 0.59], randomize=False)
    assert d.get_dataset_size() == 10
    assert d1.get_dataset_size() == 4
    assert d2.get_dataset_size() == 6
    num_iter = 0
    for item in d1.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        assert item['file_name'] == np.array(data[num_iter]['file_name'])
        num_iter += 1
    assert num_iter == 4
    num_iter = 0
    for item in d2.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        assert item['file_name'] == np.array(data[num_iter + 4]['file_name'])
        num_iter += 1
    assert num_iter == 6


def test_cv_minddataset_split_deterministic(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset after deterministic Split op is applied
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    d = ds.MindDataset(file_name + "0", columns_list,
                       num_readers, shuffle=False)
    # should set seed to avoid data overlap
    original_seed = config_get_set_seed(111)
    d1, d2 = d.split([0.8, 0.2])
    assert d.get_dataset_size() == 10
    assert d1.get_dataset_size() == 8
    assert d2.get_dataset_size() == 2

    d1_dataset = []
    d2_dataset = []
    num_iter = 0
    for item in d1.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        d1_dataset.append(item['file_name'])
        num_iter += 1
    assert num_iter == 8
    num_iter = 0
    for item in d2.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        d2_dataset.append(item['file_name'])
        num_iter += 1
    assert num_iter == 2
    inter_dataset = [x for x in d1_dataset if x in d2_dataset]
    assert inter_dataset == []  # intersection of  d1 and d2
    ds.config.set_seed(original_seed)


def test_cv_minddataset_split_sharding(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read MindDataset with DistributedSampler after deterministic Split op is applied
    Expectation: Output is equal to the expected output
    """
    os.environ["MS_DEV_MINDRECORD_SHARD_BY_BLOCK"] = "true"
    data = get_data(CV_DIR_NAME, True)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    d = ds.MindDataset(file_name + "0", columns_list,
                       num_readers, shuffle=False)
    # should set seed to avoid data overlap
    original_seed = config_get_set_seed(111)
    d1, d2 = d.split([0.8, 0.2])
    assert d.get_dataset_size() == 10
    assert d1.get_dataset_size() == 8
    assert d2.get_dataset_size() == 2
    distributed_sampler = ds.DistributedSampler(2, 0)
    d1.use_sampler(distributed_sampler)
    assert d1.get_dataset_size() == 4

    num_iter = 0
    d1_shard1 = []
    for item in d1.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
        d1_shard1.append(item['file_name'])
    assert num_iter == 4
    assert d1_shard1 != [x['file_name'] for x in data[0:4]]

    distributed_sampler = ds.DistributedSampler(2, 1)
    d1.use_sampler(distributed_sampler)
    assert d1.get_dataset_size() == 4

    d1s = d1.repeat(3)
    epoch1_dataset = []
    epoch2_dataset = []
    epoch3_dataset = []
    num_iter = 0
    for item in d1s.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
        if num_iter <= 4:
            epoch1_dataset.append(item['file_name'])
        elif num_iter <= 8:
            epoch2_dataset.append(item['file_name'])
        else:
            epoch3_dataset.append(item['file_name'])
    assert len(epoch1_dataset) == 4
    assert len(epoch2_dataset) == 4
    assert len(epoch3_dataset) == 4
    inter_dataset = [x for x in d1_shard1 if x in epoch1_dataset]
    assert inter_dataset == []  # intersection of d1's shard1 and d1's shard2
    assert epoch1_dataset not in (epoch2_dataset, epoch3_dataset)
    assert epoch2_dataset not in (epoch1_dataset, epoch3_dataset)
    assert epoch3_dataset not in (epoch1_dataset, epoch2_dataset)

    epoch1_dataset.sort()
    epoch2_dataset.sort()
    epoch3_dataset.sort()
    assert epoch1_dataset != epoch2_dataset
    assert epoch2_dataset != epoch3_dataset
    assert epoch3_dataset != epoch1_dataset

    ds.config.set_seed(original_seed)
    del os.environ["MS_DEV_MINDRECORD_SHARD_BY_BLOCK"]


def get_data(dir_name, sampler=False):
    """
    usage: get data from imagenet dataset
    params:
    dir_name: directory containing folder images and annotation information

    """
    if not os.path.isdir(dir_name):
        raise IOError("Directory {} not exists".format(dir_name))
    img_dir = os.path.join(dir_name, "images")
    if sampler:
        ann_file = os.path.join(dir_name, "annotation_sampler.txt")
    else:
        ann_file = os.path.join(dir_name, "annotation.txt")
    with open(ann_file, "r", encoding='utf-8') as file_reader:
        lines = file_reader.readlines()

    data_list = []
    for i, line in enumerate(lines):
        try:
            filename, label = line.split(",")
            label = label.strip("\n")
            with open(os.path.join(img_dir, filename), "rb") as file_reader:
                img = file_reader.read()
            data_json = {"id": i,
                         "file_name": filename,
                         "data": img,
                         "label": int(label)}
            data_list.append(data_json)
        except FileNotFoundError:
            continue
    return data_list


def check_pksampler(file_name, col_type):
    """check the PKSampler with type string, int, float"""
    if os.path.exists("{}".format(file_name)):
        os.remove("{}".format(file_name))
    if os.path.exists("{}.db".format(file_name)):
        os.remove("{}.db".format(file_name))

    if col_type == "string":
        schema_json = {"file_name": {"type": "string"}, "label": {"type": "string"}, "data": {"type": "bytes"}}
    elif col_type == "int32":
        schema_json = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"}}
    elif col_type == "int64":
        schema_json = {"file_name": {"type": "string"}, "label": {"type": "int64"}, "data": {"type": "bytes"}}
    elif col_type == "float32":
        schema_json = {"file_name": {"type": "string"}, "label": {"type": "float32"}, "data": {"type": "bytes"}}
    elif col_type == "float64":
        schema_json = {"file_name": {"type": "string"}, "label": {"type": "float64"}, "data": {"type": "bytes"}}
    else:
        raise RuntimeError("Parameter {} error".format(col_type))


    writer = FileWriter(file_name=file_name, shard_num=1, overwrite=True)
    _ = writer.add_schema(schema_json, "test_schema")
    indexes = ["file_name", "label"]
    _ = writer.add_index(indexes)
    for i in range(1000):
        if col_type == "string":
            data = [{"file_name": str(i) + ".jpg", "label": str(int(i / 100)),
                     "data": b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff"}]
        elif col_type == "int32" or col_type == "int64":
            data = [{"file_name": str(i) + ".jpg", "label": int(i / 100),
                     "data": b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff"}]
        elif col_type == "float32" or col_type == "float64":
            data = [{"file_name": str(i) + ".jpg", "label": float(int(i / 100) + 0.5),
                     "data": b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff"}]
        else:
            raise RuntimeError("Parameter {} error".format(col_type))
        _ = writer.write_raw_data(data)
    _ = writer.commit()

    sampler = ds.PKSampler(5, class_column='label')
    data_set = ds.MindDataset(dataset_files=file_name, sampler=sampler)
    assert data_set.get_dataset_size() == 50

    count = 0
    for item in data_set.create_dict_iterator(output_numpy=True):
        print("item name:", item["label"].dtype, item["label"])
        if col_type == "string":
            assert item["label"].dtype == np.array("9").dtype
        elif col_type == "int32":
            assert item["label"].dtype == np.int32
        elif col_type == "int64":
            assert item["label"].dtype == np.int64
        elif col_type == "float32":
            assert item["label"].dtype == np.float32
        elif col_type == "float64":
            assert item["label"].dtype == np.float64
        else:
            raise RuntimeError("Parameter {} error".format(col_type))
        count += 1
    assert count == 50

    if os.path.exists("{}".format(file_name)):
        os.remove("{}".format(file_name))
    if os.path.exists("{}.db".format(file_name)):
        os.remove("{}.db".format(file_name))


def test_cv_minddataset_pksampler_with_diff_type():
    """
    Feature: MindDataset
    Description: Test read MindDataset with PKSampler and use string, int, float type
    Expectation: Output is equal to the expected output
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    check_pksampler(file_name, "string")
    check_pksampler(file_name, "int32")
    check_pksampler(file_name, "int64")
    check_pksampler(file_name, "float32")
    check_pksampler(file_name, "float64")


def test_minddataset_getitem_random_sampler(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test MindDataset's __getitem__ method with RandomSampler
    Expectation: Output is equal to the expected output
    """
    origin_seed = ds.config.get_seed()
    ds.config.set_seed(1234)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.RandomSampler()
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              sampler=sampler)
    assert data_set.get_dataset_size() == 10
    num_iter = 0
    for item in data_set.create_tuple_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
        assert item == data_set[num_iter - 1]
    ds.config.set_seed(origin_seed)


@pytest.mark.parametrize('num_samples', (None, 8))
@pytest.mark.parametrize('shuffle', (True, False))
def test_minddataset_getitem_shuffle_num_samples(add_and_remove_cv_file, shuffle, num_samples):
    """
    Feature: MindDataset
    Description: Test MindDataset's __getitem__ method with shuffle and num_samples
    Expectation: Output is equal to the expected output
    """
    origin_seed = ds.config.get_seed()
    ds.config.set_seed(1234)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              shuffle=shuffle, num_samples=num_samples)
    num_iter = 0
    origin1 = []
    for item in data_set.create_tuple_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
        origin1.append(item)
        assert item == data_set[num_iter - 1]

    # Verify dataset random access followed by iterator access
    assert origin1[0] == data_set[0]

    origin2 = []
    num_iter2 = 0
    for item in data_set.create_tuple_iterator(num_epochs=1, output_numpy=True):
        num_iter2 += 1
        origin2.append(item)
        assert item == data_set[num_iter2 - 1]

    # Verify that the results of two iterator accesses to dataset are consistent
    assert origin1 == origin2
    ds.config.set_seed(origin_seed)


@pytest.mark.parametrize('shuffle', (True, False))
def test_minddataset_getitem_shuffle_distributed_sampler(add_and_remove_cv_file, shuffle):
    """
    Feature: MindDataset
    Description: Test MindDataset's __getitem__ method with shuffle and num_samples=8 and DistributedSampler(2, 2)
    Expectation: Output is equal to the expected output
    """
    origin_seed = ds.config.get_seed()
    ds.config.set_seed(1234)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              shuffle=shuffle, num_samples=8)
    sampler = ds.DistributedSampler(2, 2)
    data_set.add_sampler(sampler)
    num_iter = 0
    for item in data_set.create_tuple_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
        assert item == data_set[num_iter - 1]
    ds.config.set_seed(origin_seed)


@pytest.mark.parametrize('shuffle', (True, False))
def test_minddataset_getitem_distributed_sampler(add_and_remove_cv_file, shuffle):
    """
    Feature: MindDataset
    Description: Test MindDataset's __getitem__ method with num_samples=8 and DistributedSampler(2, 2, shuffle=shuffle)
    Expectation: Output is equal to the expected output
    """
    origin_seed = ds.config.get_seed()
    ds.config.set_seed(1234)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              shuffle=False, num_samples=8)
    sampler = ds.DistributedSampler(2, 2, shuffle=shuffle)
    data_set.add_sampler(sampler)
    num_iter = 0
    for item in data_set.create_tuple_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
        assert item == data_set[num_iter - 1]
    ds.config.set_seed(origin_seed)


def test_minddataset_getitem_random_sampler_and_distributed_sampler(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test MindDataset's __getitem__ method with RandomSampler and DistributedSampler
    Expectation: Output is equal to the expected output
    """
    origin_seed = ds.config.get_seed()
    ds.config.set_seed(1234)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.RandomSampler()
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              sampler=sampler)
    sampler = ds.DistributedSampler(2, 2)
    data_set.add_sampler(sampler)
    assert data_set.get_dataset_size() == 5
    num_iter = 0
    for item in data_set.create_tuple_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
        assert item == data_set[num_iter - 1]
    ds.config.set_seed(origin_seed)


def test_minddataset_getitem_exception(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test MindDataset's __getitem__ method with exception
    Expectation: Output is equal to the expected output
    """
    origin_seed = ds.config.get_seed()
    ds.config.set_seed(1234)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.RandomSampler()
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              sampler=sampler)
    assert data_set.get_dataset_size() == 10
    with pytest.raises(TypeError) as err:
        _ = data_set["2"]
    assert "Argument index with value 2 is not of type [<class 'int'>], but got <class 'str'>." in str(err.value)

    with pytest.raises(RuntimeError) as err2:
        _ = data_set[100]
    assert "Input index is not within the required interval of [0, 9], but got 100." in str(err2.value)

    with pytest.raises(ValueError) as err3:
        _ = data_set[-1]
    assert "index cannot be negative, but got -1." in str(err3.value)
    ds.config.set_seed(origin_seed)


@pytest.mark.parametrize("cleanup_tmp_file", ["test_distributed.mindrecord*"], indirect=True)
def test_minddataset_distributed_samples(cleanup_tmp_file):
    """
    Feature: MindDataset
    Description: Test MindDataset sharding sampling: two strategies of block sampling and shard sampling results
    Expectation: Output is equal to the expected output
    """
    origin_seed = ds.config.get_seed()
    ds.config.set_seed(1024)
    mindrecord_name = "test_distributed.mindrecord"
    writer = FileWriter(file_name=mindrecord_name, shard_num=1, overwrite=True)
    schema_json = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "float64"}}
    writer.add_schema(schema_json, "test_schema")
    indexes = ["file_name", "data"]
    writer.add_index(indexes)
    for i in range(12):
        data = [{"file_name": str(i) + ".jpg", "label": i, "data": float(i)}]
        writer.write_raw_data(data)
    writer.commit()

    columns_list = ["label", "file_name", "data"]

    # Validate the results of two strategies for shard sampling in MindDataset
    sampler1 = ds.DistributedSampler(3, 1, shuffle=False)
    output1 = [1.0, 4.0, 7.0, 10.0]
    output2 = [4.0, 5.0, 6.0, 7.0]
    os.environ["MS_DEV_MINDRECORD_SHARD_BY_BLOCK"] = "false"
    data_set1 = ds.MindDataset(mindrecord_name, columns_list, 1, sampler=sampler1)
    for i, item in enumerate(data_set1.create_dict_iterator(num_epochs=1, output_numpy=True)):
        assert output1[i] == item['data']
    data_set11 = ds.MindDataset(mindrecord_name, columns_list, 1, num_shards=3, shard_id=1, shuffle=False)
    for i, item in enumerate(data_set11.create_dict_iterator(num_epochs=1, output_numpy=True)):
        assert output1[i] == item['data']

    os.environ["MS_DEV_MINDRECORD_SHARD_BY_BLOCK"] = "true"
    data_set2 = ds.MindDataset(mindrecord_name, columns_list, 1, sampler=sampler1)
    for i, item in enumerate(data_set2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        assert output2[i] == item['data']
    data_set22 = ds.MindDataset(mindrecord_name, columns_list, 1, num_shards=3, shard_id=1, shuffle=False)
    for i, item in enumerate(data_set22.create_dict_iterator(num_epochs=1, output_numpy=True)):
        assert output2[i] == item['data']

    # Validate the results of two strategies for shard sampling in MindDataset mixed sampling
    sampler2 = ds.RandomSampler()
    output3 = [5.0, 7.0, 2.0]
    output4 = [7.0, 9.0, 6.0]
    os.environ["MS_DEV_MINDRECORD_SHARD_BY_BLOCK"] = "false"
    data_set3 = ds.MindDataset(mindrecord_name, columns_list, 1, sampler=sampler2)
    output_random_sampler = [8.0, 3.0, 5.0, 0.0, 10.0, 4.0, 7.0, 9.0, 6.0, 11.0, 2.0, 1.0]
    for i, item in enumerate(data_set3.create_dict_iterator(num_epochs=1, output_numpy=True)):
        assert output_random_sampler[i] == item['data']
    sampler3 = ds.DistributedSampler(4, 2, shuffle=False)
    data_set3.add_sampler(sampler3)
    for i, item in enumerate(data_set3.create_dict_iterator(num_epochs=1, output_numpy=True)):
        assert output3[i] == item['data']

    os.environ["MS_DEV_MINDRECORD_SHARD_BY_BLOCK"] = "true"
    data_set4 = ds.MindDataset(mindrecord_name, columns_list, 1, sampler=sampler2)
    sampler4 = ds.DistributedSampler(4, 2, shuffle=False)
    data_set4.add_sampler(sampler4)
    for i, item in enumerate(data_set4.create_dict_iterator(num_epochs=1, output_numpy=True)):
        assert output4[i] == item['data']

    # Verify MindDataset by slice sampling && num_padded parameter
    os.environ["MS_DEV_MINDRECORD_SHARD_BY_BLOCK"] = "false"
    padded_sample = {}
    padded_sample["data"] = 1234.1234
    padded_sample["file_name"] = "1234.1234.jpg"
    padded_sample["label"] = -1
    output5 = [3.0, 8.0, 1234.1234]
    data_set5 = ds.MindDataset(mindrecord_name, columns_list, 1, padded_sample=padded_sample, num_padded=3,
                               num_shards=5, shard_id=3, shuffle=False)
    for i, item in enumerate(data_set5.create_dict_iterator(num_epochs=1, output_numpy=True)):
        assert output5[i] == item['data']

    # Verify that MindDataset iterates data normally without shard sampling
    data_set6 = ds.MindDataset(mindrecord_name, columns_list, 1, shuffle=Shuffle.PARTIAL)
    count = 0
    for _ in data_set6.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == data_set6.get_dataset_size()

    ds.config.set_seed(origin_seed)
    del os.environ["MS_DEV_MINDRECORD_SHARD_BY_BLOCK"]


def apply_func(dataset):
    '''apply_func'''
    rescale = 2.0
    shift = 1.0
    meanr = 0.5
    meang = 115.0
    meanb = 100.0
    stdr = 70.0
    stdg = 68.0
    stdb = 71.0

    random_horizon = vision.RandomHorizontalFlip()
    dataset = dataset.map(input_columns="data", operations=random_horizon, num_parallel_workers=2)

    random_vertical = vision.RandomVerticalFlip()
    dataset = dataset.map(input_columns="data", operations=random_vertical, num_parallel_workers=2)

    rescale_op = vision.Rescale(rescale, shift)
    dataset = dataset.map(input_columns="data", operations=rescale_op, num_parallel_workers=2)

    normalize_op = vision.Normalize((meanr, meang, meanb), (stdr, stdg, stdb))
    dataset = dataset.map(input_columns=["data"], operations=normalize_op, num_parallel_workers=2)

    return dataset


def add_one_by_batch_num(batch_info):
    return batch_info.get_batch_num() + 1


def add_one_by_epoch(batch_info):
    return batch_info.get_epoch_num() + 1


def invert_sign_per_batch_multi_col(col_list, batch_info):
    return ([np.copy(((-1) ** batch_info.get_batch_num()) * arr) for arr in col_list],)


class UserCallback(DSCallback):
    def __init__(self, py_op, step_size=1):
        super().__init__(step_size)
        self.py_op = py_op

    def ds_step_begin(self, ds_run_context):
        ep_num = ds_run_context.cur_epoch_num
        step_num = ds_run_context.cur_step_num
        self.py_op.update(ep_num, step_num)


class UserPyOp:
    '''userpyop'''

    def __init__(self):
        self.ep_num = 0
        self.step_num = 0

    def __call__(self, x):
        if 'bytes_' in str(type(x.dtype)) or 'S' in str(x.dtype):
            x = np.frombuffer(x, dtype=np.uint8)
            x = np.array(x + self.step_num ** self.ep_num - 1)
            return np.array([x.tobytes()])
        return np.array(x + self.step_num ** self.ep_num - 1)

    def update(self, ep_num, step_num):
        self.ep_num = ep_num
        self.step_num = step_num


def dataset_call_c_transforms_func(sampler, shard_id=0, usesample=False, is_numpy_bytes=False):
    """
    All of c_transforms
    Returns:

    """

    def filter_func_ge(data, _):
        if data[0] == 137:
            return False
        return True

    def flat_map_func(x):
        d = ds.MindDataset(MINDRECORD_IMAGENET)
        return d

    crop_height = 300
    crop_width = 300
    target_height = 200
    target_width = 200
    interpolation_mode = Inter.BILINEAR
    scalelb = 0.5
    scaleub = 200.5
    aspectlb = 200.5
    aspectub = 200.5
    targetheight = 100
    targetwidth = 100
    interpolation = Inter.BILINEAR
    maxiter = 100
    skipcount = 5
    takecount = 3
    repeatcount = 10
    batchsize = 8
    l1 = []

    data = ds.TextFileDataset(DATA_FILE, num_samples=1)
    data = data.flat_map(flat_map_func)

    count = 0
    for _ in data.create_dict_iterator(output_numpy=True):
        count += 1
    assert count == 20

    op1 = UserPyOp()
    cb1 = UserCallback(op1)
    dataset = ds.MindDataset(MINDRECORD_IMAGENET)
    dataset = dataset.map(operations=op1, callbacks=cb1)
    dataset = dataset.concat(data)
    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        i += 1
    assert i == 40

    prepend_tensor = np.array([4, 2], dtype=np.uint8)
    append_tensor = np.array([9, 10], dtype=np.uint8)
    concatenate_op = transforms.Concatenate(0, prepend_tensor, append_tensor)
    dataset = dataset.map(input_columns=["image"], operations=concatenate_op)

    fill_op = transforms.Fill(-3)
    dataset = dataset.map(input_columns=["image"], operations=fill_op)
    dataset = dataset.map(input_columns=["image"], operations=transforms.Mask(transforms.Relational.EQ, 255))
    dataset = dataset.map(input_columns=["image"], operations=transforms.Slice(slice(0, 3)))

    dataset = ds.MindDataset(MINDRECORD_IMAGENET, sampler=sampler, columns_list=["data", "label"],
                             num_parallel_workers=3, shuffle=None)
    if not is_numpy_bytes:
        image_data = np.fromfile(image_jpg, dtype=np.uint8)
    else:
        image_data = np.array([np.fromfile(image_jpg, dtype=np.uint8).tobytes()])
    padded_samples = [{'data': image_data, 'label': np.array(1, np.int64)}]
    padded_ds = ds.PaddedDataset(padded_samples)
    dataset = dataset + padded_ds
    if usesample:
        testsampler = ds.DistributedSampler(num_shards=2, shard_id=shard_id, shuffle=False)
        dataset.use_sampler(testsampler)

    beforesize = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        beforesize += 1
    dataset = dataset.skip(count=skipcount)
    aftersize = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        aftersize += 1
    assert (beforesize - aftersize) == skipcount

    dataset = dataset.filter(predicate=filter_func_ge, input_columns=["data", "label"], num_parallel_workers=4)

    decode_op = vision.Decode()
    dataset = dataset.map(input_columns="data", operations=decode_op, num_parallel_workers=8)

    randomcrop_op = vision.RandomCrop(size=(crop_height, crop_width), padding=(1, 1), pad_if_needed=True,
                                       fill_value=(1, 1, 0), padding_mode=Border.CONSTANT)
    dataset = dataset.map(input_columns="data", operations=randomcrop_op, num_parallel_workers=3)

    randomcrop_op = vision.RandomCrop(size=(crop_height, crop_width), padding=(1, 1), pad_if_needed=True,
                                       fill_value=(1, 1, 0), padding_mode=Border.EDGE)
    dataset = dataset.map(input_columns="data", operations=randomcrop_op, num_parallel_workers=3)

    randomcrop_op = vision.RandomCrop(size=(crop_height, crop_width), padding=(1, 1), pad_if_needed=True,
                                       fill_value=(1, 1, 0), padding_mode=Border.REFLECT)
    dataset = dataset.map(input_columns="data", operations=randomcrop_op, num_parallel_workers=3)

    randomcrop_op = vision.RandomCrop(size=(crop_height, crop_width), padding=(1, 1), pad_if_needed=True,
                                       fill_value=(1, 1, 0), padding_mode=Border.SYMMETRIC)
    dataset = dataset.map(input_columns="data", operations=randomcrop_op, num_parallel_workers=3)

    dataset = dataset.apply(apply_func)

    resize_op = vision.Resize((target_height, target_width), interpolation_mode)
    dataset = dataset.map(input_columns="data", operations=resize_op, num_parallel_workers=3)

    randomcropandresize_op = vision.RandomResizedCrop((targetheight, targetwidth), (scalelb, scaleub),
                                                       (aspectlb, aspectub), interpolation, maxiter)
    dataset = dataset.map(input_columns=["data"], operations=randomcropandresize_op, num_parallel_workers=3)

    num_classes = dataset.num_classes()
    num_classes = 823
    one_hot_encode = transforms.OneHot(num_classes)
    dataset = dataset.map(input_columns="label", operations=one_hot_encode, num_parallel_workers=3)
    cutmix_batch_op = vision.CutMixBatch(mode.ImageBatchFormat.NHWC)
    dataset = dataset.batch(2, drop_remainder=True)
    dataset = dataset.map(input_columns=["data", "label"], operations=cutmix_batch_op)

    pad_shape = [2, 100, 100, 4]
    pad_value = -1
    dataset = dataset.map(input_columns="data", operations=transforms.PadEnd(pad_shape, pad_value))

    dataset = dataset.take(count=takecount)
    aftersize = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        aftersize += 1
    assert aftersize == takecount

    dataset = dataset.shuffle(2)
    dataset = dataset.padded_batch(batchsize, True, num_parallel_workers=3, pad_info={"image": (None, 2)})
    dataset = dataset.repeat(repeatcount)
    column_names = ["label"]
    bucket_boundaries = [1, 2, 3]
    bucket_batch_sizes = [3, 3, 2, 2]
    dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries, bucket_batch_sizes)

    unique_op = transforms.Unique()
    dataset = dataset.map(operations=unique_op, input_columns='data',
                          output_columns=['data', 'data_idx', 'data_cnt'],
                          num_parallel_workers=3)
    dataset = dataset.project(columns=['data', 'data_idx', 'data_cnt'])
    for data in dataset.create_dict_iterator(output_numpy=True):
        l1.append(data['image'])
    l1.clear()

    dataset_1 = ds.MindDataset(MINDRECORD_IMAGENET, sampler=sampler, num_parallel_workers=3)
    input_columns = ['file_name', 'label', 'data']
    output_columns = ['a', 'b', 'c']
    dataset_1 = dataset_1.rename(input_columns, output_columns)
    dataset_2 = ds.MindDataset(MINDRECORD_IMAGENET)
    dataset_zip = ds.zip((dataset_1, dataset_2))
    for data in dataset_zip.create_dict_iterator(output_numpy=True):
        l1.append(data['data'])
    l1.clear()

    dataset = ds.MindDataset(MINDRECORD_IMAGENET, sampler=sampler, num_parallel_workers=3)
    dataset = dataset.map(input_columns='data', operations=vision.Decode(), num_parallel_workers=8)

    randomrotation_op = vision.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
    dataset = dataset.map(input_columns='data', operations=randomrotation_op, num_parallel_workers=3)

    randomrotation_op = vision.RandomSharpness((0.1, 1.9))
    dataset = dataset.map(input_columns='data', operations=randomrotation_op, num_parallel_workers=3)

    randomrotation_op = vision.RandomColor((0.1, 1.9))
    dataset = dataset.map(input_columns='data', operations=randomrotation_op, num_parallel_workers=3)

    randomrotation_op = vision.RandomPosterize((1, 8))
    dataset = dataset.map(input_columns='data', operations=randomrotation_op, num_parallel_workers=3)

    randomrotation_op = vision.RandomSolarize((0, 255))
    dataset = dataset.map(input_columns='data', operations=randomrotation_op, num_parallel_workers=3)

    pad_op = vision.AutoContrast(cutoff=10.0, ignore=[10, 20])
    dataset = dataset.map(input_columns='data', operations=pad_op, num_parallel_workers=3)

    pad_op = vision.Equalize()
    dataset = dataset.map(input_columns='data', operations=pad_op, num_parallel_workers=3)

    pad_op = vision.Invert()
    dataset = dataset.map(input_columns='data', operations=pad_op, num_parallel_workers=3)

    op_list = [
        vision.CenterCrop(1),
        vision.Pad(padding=(2, 2), padding_mode=Border.EDGE),
    ]
    operations = transforms.Compose(op_list)
    dataset = dataset.map(input_columns="data", operations=operations, num_parallel_workers=3)

    # Pad: number of channels for input tensor can only be 1 or 3
    op_list = [
        vision.Pad(padding=(2, 2), padding_mode=Border.CONSTANT),
        vision.Pad(padding=(2, 2), padding_mode=Border.EDGE)
    ]
    operations = transforms.RandomApply(op_list)
    dataset = dataset.map(input_columns="data", operations=operations, num_parallel_workers=3)

    op_list = [
        vision.Pad(padding=(2, 2), padding_mode=Border.EDGE),
        vision.Pad(padding=(2, 2), padding_mode=Border.REFLECT)
    ]
    operations = transforms.RandomChoice(op_list)
    dataset = dataset.map(input_columns="data", operations=operations, num_parallel_workers=3)

    pad_op = vision.Pad(padding=(2, 2), padding_mode=Border.SYMMETRIC)
    dataset = dataset.map(input_columns="data", operations=pad_op, num_parallel_workers=3)

    randomcoloradjust_op = vision.RandomColorAdjust(brightness=(1.0, 1.0), contrast=(1, 1), saturation=(1, 1),
                                                     hue=(0, 0))
    dataset = dataset.map(input_columns="data", operations=randomcoloradjust_op, num_parallel_workers=3)

    hwc2chw_op = vision.HWC2CHW()
    dataset = dataset.map(input_columns='data', operations=hwc2chw_op, num_parallel_workers=3)

    for data in dataset.create_dict_iterator(output_numpy=True):
        l1.append(data['data'])
    l1.clear()

    dataset = ds.MindDataset(MINDRECORD_IMAGENET, sampler=sampler, num_parallel_workers=3)
    randomcropdecoderesize_op = vision.RandomCropDecodeResize(size=(20, 20), scale=(0.08, 1.0),
                                                               ratio=(0.75, 1.3333333333333333),
                                                               interpolation=Inter.BILINEAR, max_attempts=10)
    dataset = dataset.map(input_columns="data", operations=randomcropdecoderesize_op, num_parallel_workers=3)
    for data in dataset.create_dict_iterator(output_numpy=True):
        l1.append(data['data'])
    l1.clear()

    dataset = ds.MindDataset(MINDRECORD_IMAGENET, sampler=sampler, num_parallel_workers=3)
    randomcropdecoderesize_op = vision.RandomCropDecodeResize(size=(20, 20), scale=(0.08, 1.0),
                                                               ratio=(0.75, 1.3333333333333333),
                                                               interpolation=Inter.NEAREST, max_attempts=10)
    dataset.map(input_columns="data", operations=randomcropdecoderesize_op, num_parallel_workers=3)

    dataset = ds.MindDataset(MINDRECORD_IMAGENET, sampler=sampler, num_parallel_workers=3)
    randomcropdecoderesize_op = vision.RandomCropDecodeResize(size=(20, 20), scale=(0.08, 1.0),
                                                               ratio=(0.75, 1.3333333333333333),
                                                               interpolation=Inter.BICUBIC, max_attempts=10)
    dataset = dataset.map(input_columns="data", operations=randomcropdecoderesize_op, num_parallel_workers=3)

    randomresize_op = vision.RandomResize((15, 15))
    dataset = dataset.map(input_columns="data", operations=randomresize_op, num_parallel_workers=3)

    randomrotation_op = vision.RandomRotation(degrees=(0, 125), resample=Inter.BILINEAR, expand=False,
                                               center=(6, 6), fill_value=1)
    dataset = dataset.map(input_columns="data", operations=randomrotation_op, num_parallel_workers=3)

    randomrotation_op = vision.RandomRotation(degrees=(0, 125), resample=Inter.NEAREST, expand=False,
                                               center=(6, 6), fill_value=1)
    dataset = dataset.map(input_columns="data", operations=randomrotation_op, num_parallel_workers=3)

    randomrotation_op = vision.RandomRotation(degrees=(0, 125), resample=Inter.BICUBIC, expand=False,
                                               center=(6, 6), fill_value=1)
    dataset = dataset.map(input_columns="data", operations=randomrotation_op, num_parallel_workers=3)

    typecast_op = transforms.TypeCast(data_type=mstype.int8)
    dataset = dataset.map(input_columns="data", operations=typecast_op, num_parallel_workers=3)

    columns_to_project = ["data", "label"]
    dataset = dataset.project(columns=columns_to_project)

    dataset.device_que()
    l1.clear()

    dataset.create_tuple_iterator()

    dict_iterator = dataset.create_dict_iterator(output_numpy=True)
    for data in dict_iterator:
        l1.append(list(data))
    l1.clear()

    output_shape_list = dataset.output_shapes()
    for data_shape in output_shape_list:
        l1.append(data_shape)
    l1.clear()

    output_type_list = dataset.output_types()
    for data_type in output_type_list:
        l1.append(data_type)
    l1.clear()

    dataset.get_dataset_size()

    dataset.get_batch_size()

    dataset.get_repeat_count()

    dataset.num_classes()

    dataset.reset()


class DatasetCTransformsFunc():
    '''DatasetCTransformsFunc'''

    def cutmix_fun(self, sampler):
        '''cutmix_fun'''

        def filter_func_ge(data, _):
            if data[0] == 137:
                return False
            return True

        crop_height = 300
        crop_width = 300
        target_height = 200
        target_width = 200
        interpolation_mode = Inter.BILINEAR
        scalelb = 0.5
        scaleub = 200.5
        aspectlb = 200.5
        aspectub = 200.5
        targetheight = 100
        targetwidth = 100
        interpolation = Inter.BILINEAR
        maxiter = 100
        takecount = 3
        repeatcount = 10
        batchsize = 8
        l1 = []

        dataset = ds.MindDataset(MINDRECORD_IMAGENET, sampler=sampler, columns_list=["data", "label"],
                                 num_parallel_workers=3, shuffle=None)

        dataset = dataset.filter(predicate=filter_func_ge, input_columns=["data", "label"], num_parallel_workers=4)

        decode_op = vision.Decode()
        dataset = dataset.map(input_columns="data", operations=decode_op, num_parallel_workers=3)

        randomcrop_op = vision.RandomCrop(size=(crop_height, crop_width), padding=(1, 1), pad_if_needed=True,
                                           fill_value=(1, 1, 0), padding_mode=Border.CONSTANT)
        dataset = dataset.map(input_columns="data", operations=randomcrop_op, num_parallel_workers=3)

        randomcrop_op = vision.RandomCrop(size=(crop_height, crop_width), padding=(1, 1), pad_if_needed=True,
                                           fill_value=(1, 1, 0), padding_mode=Border.EDGE)
        dataset = dataset.map(input_columns="data", operations=randomcrop_op, num_parallel_workers=3)

        randomcrop_op = vision.RandomCrop(size=(crop_height, crop_width), padding=(1, 1), pad_if_needed=True,
                                           fill_value=(1, 1, 0), padding_mode=Border.REFLECT)
        dataset = dataset.map(input_columns="data", operations=randomcrop_op, num_parallel_workers=3)

        randomcrop_op = vision.RandomCrop(size=(crop_height, crop_width), padding=(1, 1), pad_if_needed=True,
                                           fill_value=(1, 1, 0), padding_mode=Border.SYMMETRIC)
        dataset = dataset.map(input_columns="data", operations=randomcrop_op, num_parallel_workers=3)

        dataset = dataset.apply(apply_func)

        resize_op = vision.Resize((target_height, target_width), interpolation_mode)
        dataset = dataset.map(input_columns="data", operations=resize_op, num_parallel_workers=3)

        randomcropandresize_op = vision.RandomResizedCrop((targetheight, targetwidth), (scalelb, scaleub),
                                                           (aspectlb, aspectub), interpolation, maxiter)
        dataset = dataset.map(input_columns=["data"], operations=randomcropandresize_op, num_parallel_workers=3)

        num_classes = 823
        one_hot_encode = transforms.OneHot(num_classes)
        dataset = dataset.map(input_columns="label", operations=one_hot_encode, num_parallel_workers=3)
        cutmix_batch_op = vision.CutMixBatch(mode.ImageBatchFormat.NHWC)
        dataset = dataset.batch(2, drop_remainder=True)
        dataset = dataset.map(input_columns=["data", "label"], operations=cutmix_batch_op)

        pad_shape = [2, 100, 100, 4]
        pad_value = -1
        dataset = dataset.map(input_columns="data", operations=transforms.PadEnd(pad_shape, pad_value))

        dataset = dataset.take(count=takecount)
        dataset = dataset.shuffle(2)
        dataset = dataset.padded_batch(batchsize, True, num_parallel_workers=3, pad_info={"image": (None, 2)})
        dataset = dataset.repeat(repeatcount)
        column_names = ["label"]
        bucket_boundaries = [1, 2, 3]
        bucket_batch_sizes = [3, 3, 2, 2]
        dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries, bucket_batch_sizes)
        for data in dataset.create_dict_iterator(output_numpy=True):
            l1.append(data['image'])
        l1.clear()


def dataset_call_py_transforms_func(sampler, sampler_num):
    """
    All of py_transforms
    Returns:

    """
    crop_height = 100
    crop_width = 50
    target_height = 200
    target_width = 200
    scalelb = 0.5
    scaleub = 200.5
    aspectlb = 200.5
    aspectub = 200.5
    targetheight = 100
    targetwidth = 100
    interpolation = Inter.BILINEAR
    maxiter = 100
    transformation_matrix = np.ones([432, 432])
    mean_vector = np.ones([432])
    l1 = []

    dataset = ds.MindDataset(MINDRECORD_IMAGENET, columns_list=["data", "label"], sampler=sampler,
                             num_parallel_workers=3)
    dataset_num = 0
    for _ in dataset.create_dict_iterator(output_numpy=True):
        dataset_num += 1
    assert dataset_num == sampler_num

    op_list = [vision.RandomCrop(size=(crop_height, crop_width), padding=(1, 1), pad_if_needed=True,
                                  fill_value=(1, 1, 0), padding_mode=Border.CONSTANT),
               vision.RandomCrop(size=(crop_height, crop_width), padding=(1, 1), pad_if_needed=True,
                                  fill_value=(1, 1, 0), padding_mode=Border.EDGE),
               vision.RandomCrop(size=(crop_height, crop_width), padding=(1, 1), pad_if_needed=True,
                                  fill_value=(1, 1, 0), padding_mode=Border.REFLECT),
               vision.RandomCrop(size=(crop_height, crop_width), padding=(1, 1), pad_if_needed=True,
                                  fill_value=(1, 1, 0), padding_mode=Border.SYMMETRIC),
               vision.RandomHorizontalFlip(),
               vision.RandomVerticalFlip(),
               vision.Grayscale(3),
               vision.RandomGrayscale(0.3),
               vision.RandomPerspective(distortion_scale=0.5, prob=0.1, interpolation=Inter.BICUBIC),
               vision.RandomPerspective(distortion_scale=0.5, prob=0.1, interpolation=Inter.NEAREST),
               vision.RandomPerspective(distortion_scale=0.5, prob=0.1, interpolation=Inter.BILINEAR),
               vision.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
               vision.RandomSharpness((0.1, 1.9)),
               vision.RandomColor((0.1, 1.9)),
               vision.RandomResizedCrop((targetheight, targetwidth), (scalelb, scaleub),
                                         (aspectlb, aspectub), interpolation, maxiter),
               vision.AutoContrast(cutoff=10.0, ignore=[10, 20]),
               vision.Equalize(),
               vision.Invert()
               ]

    operations = transforms.Compose([vision.Decode(to_pil=True),
                                  vision.UniformAugment(transforms=op_list),
                                  vision.Resize((224, 224)),
                                  vision.ToTensor()])

    dataset = dataset.map(operations=operations, input_columns="data", num_parallel_workers=3,
                          python_multiprocessing=True)
    dataset = dataset.shuffle(2)

    dataset = dataset.batch(batch_size=add_one_by_batch_num, drop_remainder=True, num_parallel_workers=3,
                            input_columns=["data"], per_batch_map=invert_sign_per_batch_multi_col)

    dataset = dataset.repeat(10)

    for data in dataset.create_dict_iterator(output_numpy=True):
        l1.append(data['data'])
    l1.clear()

    dataset_1 = ds.MindDataset(MINDRECORD_IMAGENET, columns_list=["data", "label"], sampler=sampler,
                               num_parallel_workers=3)
    transform_list = [
        vision.Resize((target_height, target_width)),
        vision.CenterCrop(1),
    ]
    op_list_1 = [
        vision.Decode(to_pil=True),
        vision.Resize((target_height, target_width)),
        vision.CenterCrop(1),
        vision.Pad(padding=(2, 2), padding_mode=Border.CONSTANT),
        vision.Pad(padding=(2, 2), padding_mode=Border.EDGE),
        vision.Pad(padding=(2, 2), padding_mode=Border.REFLECT),
        vision.Pad(padding=(2, 2), padding_mode=Border.SYMMETRIC),
        vision.RandomColorAdjust(brightness=(1.0, 1.0), contrast=(1, 1), saturation=(1, 1),
                                  hue=(0, 0)),
        vision.RandomRotation(degrees=(0, 125), resample=Inter.BILINEAR, expand=False,
                               center=(6, 6), fill_value=1),
        vision.RandomRotation(degrees=(0, 125), resample=Inter.NEAREST, expand=False,
                               center=(6, 6), fill_value=1),
        vision.RandomRotation(degrees=(0, 125), resample=Inter.BICUBIC, expand=False,
                               center=(6, 6), fill_value=1),
        transforms.RandomChoice(transform_list),
        transforms.RandomApply(transform_list, prob=0.5),
        transforms.RandomOrder(transform_list),
        vision.Resize(12, interpolation),
        vision.ToTensor(),
        vision.RandomErasing(prob=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False,
                              max_attempts=10),
        vision.LinearTransformation(transformation_matrix, mean_vector)

    ]
    operations_1 = transforms.Compose(op_list_1)

    dataset_1 = dataset_1.map(operations=operations_1, input_columns="data", num_parallel_workers=8,
                              python_multiprocessing=True)
    dataset_1 = dataset_1.shuffle(2)

    dataset_1 = dataset_1.padded_batch(batch_size=add_one_by_epoch, drop_remainder=True, num_parallel_workers=3,
                                       pad_info={"label": (None, 2)})

    dataset_1 = dataset_1.repeat(10)
    for data in dataset_1.create_dict_iterator(output_numpy=True):
        l1.append(data['data'])
    l1.clear()

    dataset_2 = ds.MindDataset(MINDRECORD_IMAGENET, sampler=sampler, num_parallel_workers=3)
    op_list_2 = [
        vision.Decode(to_pil=True),
        vision.FiveCrop(size=(2, 2)),
        vision.TenCrop(size=(2, 2)),
        lambda images: np.stack([vision.ToTensor()(image) for image in images]),
        vision.ToType(np.float32),
        vision.ToPIL()
    ]
    operations_2 = transforms.Compose(op_list_2)
    dataset_2.map(operations=operations_2, input_columns=["image"], num_parallel_workers=3)
    for data in dataset_2.create_dict_iterator(output_numpy=True):
        l1.append(data["data"])
    l1.clear()


def test_func_minddataset_with_subset_random_sampler_01():
    """
    Feature: MindDataset
    Description: Test read MindDataset with SubsetRandomSampler
    Expectation: Output is equal to the expected output
    """
    indices = [0, 1, 2, 3, 7, 10, 12, 14, 16, 18, 19, 20, 25, 26, 27, 28, 29, 35]
    sampler = ds.SubsetRandomSampler(indices)
    dataset_call_c_transforms_func(sampler=sampler, shard_id=1, is_numpy_bytes=True)
    DatasetCTransformsFunc().cutmix_fun(sampler=sampler)


def test_func_minddataset_with_subset_random_sampler_02():
    """
    Feature: MindDataset
    Description: Test read MindDataset with SubsetRandomSampler
    Expectation: Output is equal to the expected output
    """
    indices = [0, 1, 2, 3, 7, 10, 12, 14, 16, 18, 19, 20, 25, 26, 27, 28, 29, 35]
    sampler = ds.SubsetRandomSampler(indices)
    dataset_call_py_transforms_func(sampler=sampler, sampler_num=18)


def test_func_minddataset_with_pk_sampler_01():
    """
    Feature: MindDataset
    Description: Test read MindDataset with PKSampler
    Expectation: Output is equal to the expected output
    """
    sampler = ds.PKSampler(num_val=3, shuffle=True)
    dataset_call_c_transforms_func(sampler=sampler, usesample=True, is_numpy_bytes=True)


def test_func_minddataset_with_pk_sampler_02():
    """
    Feature: MindDataset
    Description: Test read MindDataset with PKSampler
    Expectation: Output is equal to the expected output
    """
    sampler = ds.PKSampler(num_val=2, shuffle=True, num_samples=8)
    dataset_call_py_transforms_func(sampler=sampler, sampler_num=8)


if __name__ == '__main__':
    test_cv_minddataset_pk_sample_no_column(add_and_remove_cv_file)
    test_cv_minddataset_pk_sample_basic(add_and_remove_cv_file)
    test_cv_minddataset_pk_sample_shuffle(add_and_remove_cv_file)
    test_cv_minddataset_pk_sample_out_of_range(add_and_remove_cv_file)
    test_cv_minddataset_subset_random_sample_basic(add_and_remove_cv_file)
    test_cv_minddataset_subset_random_sample_replica(add_and_remove_cv_file)
    test_cv_minddataset_subset_random_sample_empty(add_and_remove_cv_file)
    test_cv_minddataset_subset_random_sample_out_of_range(add_and_remove_cv_file)
    test_cv_minddataset_subset_random_sample_negative(add_and_remove_cv_file)
    test_cv_minddataset_random_sampler_basic(add_and_remove_cv_file)
    test_cv_minddataset_random_sampler_repeat(add_and_remove_cv_file)
    test_cv_minddataset_random_sampler_replacement(add_and_remove_cv_file)
    test_cv_minddataset_sequential_sampler_basic(add_and_remove_cv_file)
    test_cv_minddataset_sequential_sampler_exceed_size(add_and_remove_cv_file)
    test_cv_minddataset_split_basic(add_and_remove_cv_file)
    test_cv_minddataset_split_exact_percent(add_and_remove_cv_file)
    test_cv_minddataset_split_fuzzy_percent(add_and_remove_cv_file)
    test_cv_minddataset_split_deterministic(add_and_remove_cv_file)
    test_cv_minddataset_split_sharding(add_and_remove_cv_file)
    test_cv_minddataset_pksampler_with_diff_type()
    test_minddataset_getitem_random_sampler(add_and_remove_cv_file)
    test_minddataset_getitem_shuffle_num_samples(add_and_remove_cv_file, True, None)
    test_minddataset_getitem_shuffle_distributed_sampler(add_and_remove_cv_file, True)
    test_minddataset_getitem_distributed_sampler(add_and_remove_cv_file, True)
    test_minddataset_getitem_random_sampler_and_distributed_sampler(add_and_remove_cv_file)
    test_minddataset_getitem_exception(add_and_remove_cv_file)
    test_minddataset_distributed_samples(cleanup_tmp_file)
    test_func_minddataset_with_subset_random_sampler_01()
    test_func_minddataset_with_subset_random_sampler_02()
    test_func_minddataset_with_pk_sampler_01()
    test_func_minddataset_with_pk_sampler_02()
