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
import json
import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore import log as logger

FILES = ["../data/dataset/testTFTestAllTypes/test.data"]
DATASET_ROOT = "../data/dataset/testTFTestAllTypes/"
SCHEMA_FILE = "../data/dataset/testTFTestAllTypes/datasetSchema.json"


def test_schema_simple():
    """
    Feature: Schema
    Description: Test Schema simple case
    Expectation: Runs successfully
    """
    logger.info("test_schema_simple")
    ds.Schema(SCHEMA_FILE)


def test_schema_file_vs_string():
    """
    Feature: Schema
    Description: Test Schema by comparing file and string
    Expectation: Both datasets are equal
    """
    logger.info("test_schema_file_vs_string")

    schema1 = ds.Schema(SCHEMA_FILE)
    with open(SCHEMA_FILE) as file:
        json_obj = json.load(file)
        schema2 = ds.Schema()
        schema2.from_json(json_obj)

    dataset1 = ds.TFRecordDataset(FILES, schema1, shuffle=False)
    dataset2 = ds.TFRecordDataset(FILES, schema2, shuffle=False)

    for row1, row2 in zip(dataset1.create_tuple_iterator(output_numpy=True),
                          dataset2.create_tuple_iterator(output_numpy=True)):
        for col1, col2 in zip(row1, row2):
            np.testing.assert_equal(col1, col2)


def test_schema_exception():
    """
    Feature: Schema
    Description: Test Schema with invalid inputs
    Expectation: Correct error is raised as expected
    """
    logger.info("test_schema_exception")

    with pytest.raises(TypeError) as info:
        ds.Schema(1)
    assert "path: 1 is not string" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        schema = ds.Schema(SCHEMA_FILE)
        columns = [{'type': 'int8', 'shape': [3, 3]}]
        schema.parse_columns(columns)
    assert "Column's name is missing" in str(info.value)


if __name__ == '__main__':
    test_schema_simple()
    test_schema_file_vs_string()
    test_schema_exception()
