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
import numpy as np

import mindspore.dataset as ds

DATA_DIR = ["../data/dataset/testTFTestAllTypes/test.data"]
SCHEMA_DIR = "../data/dataset/testTFTestAllTypes/datasetSchema.json"
COLUMNS = ["col_1d", "col_2d", "col_3d", "col_binary", "col_float",
           "col_sint16", "col_sint32", "col_sint64"]


def check(project_columns):
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=COLUMNS)
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=project_columns)

    for data_actual, data_expected in zip(data1.create_tuple_iterator(project_columns), data2.create_tuple_iterator()):
        assert len(data_actual) == len(data_expected)
        assert all([np.array_equal(d1, d2) for d1, d2 in zip(data_actual, data_expected)])


def test_case_iterator():
    """
    Test creating tuple iterator
    """
    check(COLUMNS)
    check(COLUMNS[0:1])
    check(COLUMNS[0:2])
    check(COLUMNS[0:7])
    check(COLUMNS[7:8])
    check(COLUMNS[0:2:8])
