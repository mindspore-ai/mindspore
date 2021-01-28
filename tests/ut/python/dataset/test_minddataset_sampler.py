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
"""
This is the test module for mindrecord
"""
import os
import pytest
import numpy as np

import mindspore.dataset as ds
from mindspore import log as logger
from mindspore.dataset.text import to_str
from mindspore.mindrecord import FileWriter

FILES_NUM = 4
CV_FILE_NAME = "../data/mindrecord/imagenet.mindrecord"
CV_DIR_NAME = "../data/mindrecord/testImageNetData"


@pytest.fixture
def add_and_remove_cv_file():
    """add/remove cv file"""
    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    try:
        for x in paths:
            if os.path.exists("{}".format(x)):
                os.remove("{}".format(x))
            if os.path.exists("{}.db".format(x)):
                os.remove("{}.db".format(x))
        writer = FileWriter(CV_FILE_NAME, FILES_NUM)
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
    else:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))


def test_cv_minddataset_pk_sample_no_column(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    num_readers = 4
    sampler = ds.PKSampler(2)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", None, num_readers,
                              sampler=sampler)

    assert data_set.get_dataset_size() == 6
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[file_name]: \
                {}------------------------".format(to_str(item["file_name"])))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1


def test_cv_minddataset_pk_sample_basic(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.PKSampler(2)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                              sampler=sampler)

    assert data_set.get_dataset_size() == 6
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[data]: \
                {}------------------------".format(item["data"][:10]))
        logger.info("-------------- item[file_name]: \
                {}------------------------".format(to_str(item["file_name"])))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1


def test_cv_minddataset_pk_sample_shuffle(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.PKSampler(3, None, True)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                              sampler=sampler)

    assert data_set.get_dataset_size() == 9
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[file_name]: \
                {}------------------------".format(to_str(item["file_name"])))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 9


def test_cv_minddataset_pk_sample_shuffle_1(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.PKSampler(3, None, True, 'label', 5)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                              sampler=sampler)

    assert data_set.get_dataset_size() == 5
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[file_name]: \
                {}------------------------".format(to_str(item["file_name"])))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 5


def test_cv_minddataset_pk_sample_shuffle_2(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.PKSampler(3, None, True, 'label', 10)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                              sampler=sampler)

    assert data_set.get_dataset_size() == 9
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[file_name]: \
                {}------------------------".format(to_str(item["file_name"])))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 9


def test_cv_minddataset_pk_sample_out_of_range_0(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.PKSampler(5, None, True)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                              sampler=sampler)
    assert data_set.get_dataset_size() == 15
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[file_name]: \
                {}------------------------".format(to_str(item["file_name"])))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 15


def test_cv_minddataset_pk_sample_out_of_range_1(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.PKSampler(5, None, True, 'label', 20)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                              sampler=sampler)
    assert data_set.get_dataset_size() == 15
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[file_name]: \
                {}------------------------".format(to_str(item["file_name"])))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 15


def test_cv_minddataset_pk_sample_out_of_range_2(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.PKSampler(5, None, True, 'label', 10)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                              sampler=sampler)
    assert data_set.get_dataset_size() == 10
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[file_name]: \
                {}------------------------".format(to_str(item["file_name"])))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 10


def test_cv_minddataset_subset_random_sample_basic(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    indices = [1, 2, 3, 5, 7]
    samplers = (ds.SubsetRandomSampler(indices), ds.SubsetSampler(indices))
    for sampler in samplers:
        data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
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
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    indices = [1, 2, 2, 5, 7, 9]
    samplers = ds.SubsetRandomSampler(indices), ds.SubsetSampler(indices)
    for sampler in samplers:
        data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
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
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    indices = []
    samplers = ds.SubsetRandomSampler(indices), ds.SubsetSampler(indices)
    for sampler in samplers:
        data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
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
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    indices = [1, 2, 4, 11, 13]
    samplers = ds.SubsetRandomSampler(indices), ds.SubsetSampler(indices)
    for sampler in samplers:
        data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
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
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    indices = [1, 2, 4, -1, -2]
    samplers = ds.SubsetRandomSampler(indices), ds.SubsetSampler(indices)
    for sampler in samplers:
        data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
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
    data = get_data(CV_DIR_NAME, True)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.RandomSampler()
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
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
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.RandomSampler()
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
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
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.RandomSampler(replacement=True, num_samples=5)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
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
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.RandomSampler(replacement=False, num_samples=2)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
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
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.RandomSampler(replacement=False, num_samples=20)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
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
    data = get_data(CV_DIR_NAME, True)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.SequentialSampler(1, 4)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
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
        assert item['file_name'] == np.array(
            data[num_iter + 1]['file_name'], dtype='S')
        num_iter += 1
    assert num_iter == 4


def test_cv_minddataset_sequential_sampler_offeset(add_and_remove_cv_file):
    data = get_data(CV_DIR_NAME, True)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.SequentialSampler(2, 10)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                              sampler=sampler)
    dataset_size = data_set.get_dataset_size()
    assert dataset_size == 10
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
        assert item['file_name'] == np.array(
            data[(num_iter + 2) % dataset_size]['file_name'], dtype='S')
        num_iter += 1
    assert num_iter == 10


def test_cv_minddataset_sequential_sampler_exceed_size(add_and_remove_cv_file):
    data = get_data(CV_DIR_NAME, True)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.SequentialSampler(2, 20)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                              sampler=sampler)
    dataset_size = data_set.get_dataset_size()
    assert dataset_size == 10
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
        assert item['file_name'] == np.array(
            data[(num_iter + 2) % dataset_size]['file_name'], dtype='S')
        num_iter += 1
    assert num_iter == 10


def test_cv_minddataset_split_basic(add_and_remove_cv_file):
    data = get_data(CV_DIR_NAME, True)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    d = ds.MindDataset(CV_FILE_NAME + "0", columns_list,
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
        assert item['file_name'] == np.array(data[num_iter]['file_name'],
                                             dtype='S')
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
        assert item['file_name'] == np.array(data[num_iter + 8]['file_name'],
                                             dtype='S')
        num_iter += 1
    assert num_iter == 2


def test_cv_minddataset_split_exact_percent(add_and_remove_cv_file):
    data = get_data(CV_DIR_NAME, True)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    d = ds.MindDataset(CV_FILE_NAME + "0", columns_list,
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
        assert item['file_name'] == np.array(
            data[num_iter]['file_name'], dtype='S')
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
        assert item['file_name'] == np.array(data[num_iter + 8]['file_name'],
                                             dtype='S')
        num_iter += 1
    assert num_iter == 2


def test_cv_minddataset_split_fuzzy_percent(add_and_remove_cv_file):
    data = get_data(CV_DIR_NAME, True)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    d = ds.MindDataset(CV_FILE_NAME + "0", columns_list,
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
        assert item['file_name'] == np.array(
            data[num_iter]['file_name'], dtype='S')
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
        assert item['file_name'] == np.array(data[num_iter + 4]['file_name'],
                                             dtype='S')
        num_iter += 1
    assert num_iter == 6


def test_cv_minddataset_split_deterministic(add_and_remove_cv_file):
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    d = ds.MindDataset(CV_FILE_NAME + "0", columns_list,
                       num_readers, shuffle=False)
    # should set seed to avoid data overlap
    ds.config.set_seed(111)
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


def test_cv_minddataset_split_sharding(add_and_remove_cv_file):
    data = get_data(CV_DIR_NAME, True)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    d = ds.MindDataset(CV_FILE_NAME + "0", columns_list,
                       num_readers, shuffle=False)
    # should set seed to avoid data overlap
    ds.config.set_seed(111)
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
    with open(ann_file, "r") as file_reader:
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
