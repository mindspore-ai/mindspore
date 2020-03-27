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

DATA_DIR = "../data/dataset/test_tf_file_3_images/data"
SCHEMA = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
COLUMNS = ["label"]
GENERATE_GOLDEN = False


def test_case_0():
    logger.info("Test 0 readdir")

    # apply dataset operations
    data1 = ds.engine.Dataset.read_dir(DATA_DIR, SCHEMA, columns_list=None, num_parallel_workers=None,
                                       deterministic_output=True, prefetch_size=None, shuffle=False, seed=None)

    i = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        logger.info("item[label] is {}".format(item["label"]))
        i = i + 1
    assert (i == 3)


def test_case_1():
    logger.info("Test 1 readdir")

    # apply dataset operations
    data1 = ds.engine.Dataset.read_dir(DATA_DIR, SCHEMA, COLUMNS, num_parallel_workers=None,
                                       deterministic_output=True, prefetch_size=None, shuffle=True, seed=None)

    i = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        logger.info("item[label] is {}".format(item["label"]))
        i = i + 1
    assert (i == 3)


def test_case_2():
    logger.info("Test 2 readdir")

    # apply dataset operations
    data1 = ds.engine.Dataset.read_dir(DATA_DIR, SCHEMA, columns_list=None, num_parallel_workers=2,
                                       deterministic_output=False, prefetch_size=16, shuffle=True, seed=10)

    i = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        logger.info("item[label] is {}".format(item["label"]))
        i = i + 1
    assert (i == 3)


if __name__ == "__main__":
    test_case_0()
    test_case_1()
    test_case_2()
