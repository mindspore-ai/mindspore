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
from util import save_and_check

import mindspore.dataset as ds
from mindspore import log as logger

DATA_DIR = ["../data/dataset/testTFTestAllTypes/test.data"]
SCHEMA_DIR = "../data/dataset/testTFTestAllTypes/datasetSchema.json"
COLUMNS = ["col_1d", "col_2d", "col_3d", "col_binary", "col_float",
           "col_sint16", "col_sint32", "col_sint64"]
GENERATE_GOLDEN = False


def test_case_storage():
    """
    test StorageDataset
    """
    logger.info("Test Simple StorageDataset")
    # define parameters
    parameters = {"params": {}}

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    filename = "storage_result.npz"
    save_and_check(data1, parameters, filename, generate_golden=GENERATE_GOLDEN)


def test_case_no_rows():
    DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
    SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetNoRowsSchema.json"

    dataset = ds.StorageDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"])
    assert dataset.get_dataset_size() == 3
    count = 0
    for data in dataset.create_tuple_iterator():
        count += 1
    assert count == 3
