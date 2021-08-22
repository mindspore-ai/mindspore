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
import json
import pytest
import mindspore.dataset as ds
from mindspore import log as logger
from util import dataset_equal

FILES = ["../data/dataset/testTFTestAllTypes/test.data"]
DATASET_ROOT = "../data/dataset/testTFTestAllTypes/"
SCHEMA_FILE = "../data/dataset/testTFTestAllTypes/datasetSchema.json"


def test_schema_simple():
    logger.info("test_schema_simple")
    ds.Schema(SCHEMA_FILE)


def test_schema_file_vs_string():
    logger.info("test_schema_file_vs_string")

    schema1 = ds.Schema(SCHEMA_FILE)
    with open(SCHEMA_FILE) as file:
        json_obj = json.load(file)
        schema2 = ds.Schema()
        schema2.from_json(json_obj)

    ds1 = ds.TFRecordDataset(FILES, schema1)
    ds2 = ds.TFRecordDataset(FILES, schema2)

    dataset_equal(ds1, ds2, 0)


def test_schema_exception():
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
