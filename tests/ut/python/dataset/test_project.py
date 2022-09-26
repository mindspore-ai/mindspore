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

from util import save_and_check_tuple

import mindspore.dataset as ds
import mindspore.dataset.transforms as C
from mindspore.common import dtype as mstype

DATA_DIR_TF = ["../data/dataset/testTFTestAllTypes/test.data"]
SCHEMA_DIR_TF = "../data/dataset/testTFTestAllTypes/datasetSchema.json"
GENERATE_GOLDEN = False


def test_case_project_single_column():
    """
    Feature: Project op
    Description: Test Project op on a single column
    Expectation: Output is equal to the expected output
    """
    columns = ["col_sint32"]
    parameters = {"params": {'columns': columns}}

    data1 = ds.TFRecordDataset(DATA_DIR_TF, SCHEMA_DIR_TF, shuffle=False)
    data1 = data1.project(columns=columns)

    filename = "project_single_column_result.npz"
    save_and_check_tuple(data1, parameters, filename, generate_golden=GENERATE_GOLDEN)


def test_case_project_multiple_columns_in_order():
    """
    Feature: Project op
    Description: Test Project op on multiple columns in order
    Expectation: Output is equal to the expected output
    """
    columns = ["col_sint16", "col_float", "col_2d"]
    parameters = {"params": {'columns': columns}}

    data1 = ds.TFRecordDataset(DATA_DIR_TF, SCHEMA_DIR_TF, shuffle=False)
    data1 = data1.project(columns=columns)

    filename = "project_multiple_columns_in_order_result.npz"
    save_and_check_tuple(data1, parameters, filename, generate_golden=GENERATE_GOLDEN)


def test_case_project_multiple_columns_out_of_order():
    """
    Feature: Project op
    Description: Test Project op on multiple columns out of order
    Expectation: Output is equal to the expected output
    """
    columns = ["col_3d", "col_sint64", "col_2d"]
    parameters = {"params": {'columns': columns}}

    data1 = ds.TFRecordDataset(DATA_DIR_TF, SCHEMA_DIR_TF, shuffle=False)
    data1 = data1.project(columns=columns)

    filename = "project_multiple_columns_out_of_order_result.npz"
    save_and_check_tuple(data1, parameters, filename, generate_golden=GENERATE_GOLDEN)


def test_case_project_map():
    """
    Feature: Project op
    Description: Test Project op followed by a Map op
    Expectation: Output is equal to the expected output
    """
    columns = ["col_3d", "col_sint64", "col_2d"]
    parameters = {"params": {'columns': columns}}

    data1 = ds.TFRecordDataset(DATA_DIR_TF, SCHEMA_DIR_TF, shuffle=False)
    data1 = data1.project(columns=columns)

    type_cast_op = C.TypeCast(mstype.int64)
    data1 = data1.map(operations=type_cast_op, input_columns=["col_3d"])

    filename = "project_map_after_result.npz"
    save_and_check_tuple(data1, parameters, filename, generate_golden=GENERATE_GOLDEN)


def test_case_map_project():
    """
    Feature: Project op
    Description: Test Project op after a Map op
    Expectation: Output is equal to the expected output
    """
    columns = ["col_3d", "col_sint64", "col_2d"]
    parameters = {"params": {'columns': columns}}

    data1 = ds.TFRecordDataset(DATA_DIR_TF, SCHEMA_DIR_TF, shuffle=False)

    type_cast_op = C.TypeCast(mstype.int64)
    data1 = data1.map(operations=type_cast_op, input_columns=["col_sint64"])

    data1 = data1.project(columns=columns)

    filename = "project_map_before_result.npz"
    save_and_check_tuple(data1, parameters, filename, generate_golden=GENERATE_GOLDEN)


def test_case_project_between_maps():
    """
    Feature: Project op
    Description: Test Project op between Map ops (Map -> Project -> Map)
    Expectation: Output is equal to the expected output
    """
    columns = ["col_3d", "col_sint64", "col_2d"]
    parameters = {"params": {'columns': columns}}

    data1 = ds.TFRecordDataset(DATA_DIR_TF, SCHEMA_DIR_TF, shuffle=False)

    type_cast_op = C.TypeCast(mstype.int64)
    data1 = data1.map(operations=type_cast_op, input_columns=["col_3d"])
    data1 = data1.map(operations=type_cast_op, input_columns=["col_3d"])
    data1 = data1.map(operations=type_cast_op, input_columns=["col_3d"])
    data1 = data1.map(operations=type_cast_op, input_columns=["col_3d"])

    data1 = data1.project(columns=columns)

    data1 = data1.map(operations=type_cast_op, input_columns=["col_3d"])
    data1 = data1.map(operations=type_cast_op, input_columns=["col_3d"])
    data1 = data1.map(operations=type_cast_op, input_columns=["col_3d"])
    data1 = data1.map(operations=type_cast_op, input_columns=["col_3d"])
    data1 = data1.map(operations=type_cast_op, input_columns=["col_3d"])

    filename = "project_between_maps_result.npz"
    save_and_check_tuple(data1, parameters, filename, generate_golden=GENERATE_GOLDEN)


def test_case_project_repeat():
    """
    Feature: Project op
    Description: Test Project op followed by Repeat op
    Expectation: Output is equal to the expected output
    """
    columns = ["col_3d", "col_sint64", "col_2d"]
    parameters = {"params": {'columns': columns}}

    data1 = ds.TFRecordDataset(DATA_DIR_TF, SCHEMA_DIR_TF, shuffle=False)
    data1 = data1.project(columns=columns)

    repeat_count = 3
    data1 = data1.repeat(repeat_count)

    filename = "project_before_repeat_result.npz"
    save_and_check_tuple(data1, parameters, filename, generate_golden=GENERATE_GOLDEN)


def test_case_repeat_project():
    """
    Feature: Project op
    Description: Test Project op after a Repeat op
    Expectation: Output is equal to the expected output
    """
    columns = ["col_3d", "col_sint64", "col_2d"]
    parameters = {"params": {'columns': columns}}

    data1 = ds.TFRecordDataset(DATA_DIR_TF, SCHEMA_DIR_TF, shuffle=False)

    repeat_count = 3
    data1 = data1.repeat(repeat_count)

    data1 = data1.project(columns=columns)

    filename = "project_after_repeat_result.npz"
    save_and_check_tuple(data1, parameters, filename, generate_golden=GENERATE_GOLDEN)


def test_case_map_project_map_project():
    """
    Feature: Project op
    Description: Test Map -> Project -> Map -> Project
    Expectation: Output is equal to the expected output
    """
    columns = ["col_3d", "col_sint64", "col_2d"]
    parameters = {"params": {'columns': columns}}

    data1 = ds.TFRecordDataset(DATA_DIR_TF, SCHEMA_DIR_TF, shuffle=False)

    type_cast_op = C.TypeCast(mstype.int64)
    data1 = data1.map(operations=type_cast_op, input_columns=["col_sint64"])

    data1 = data1.project(columns=columns)

    data1 = data1.map(operations=type_cast_op, input_columns=["col_2d"])

    data1 = data1.project(columns=columns)

    filename = "project_alternate_parallel_inline_result.npz"
    save_and_check_tuple(data1, parameters, filename, generate_golden=GENERATE_GOLDEN)


def test_project_operation():
    """
    Feature: Project op
    Description: Test Project op where the output dict should maintain the insertion order
    Expectation: Output is equal to the expected output
    """
    def gen_3_cols(num):
        for i in range(num):
            yield (np.array([i * 3]), np.array([i * 3 + 1]), np.array([i * 3 + 2]))

    def test_config(num, col_order):
        dst = ds.GeneratorDataset((lambda: gen_3_cols(num)), ["col1", "col2", "col3"]).batch(batch_size=num)
        dst = dst.project(col_order)
        res = dict()
        for item in dst.create_dict_iterator(num_epochs=1):
            res = item
        return res

    assert list(test_config(1, ["col3", "col2", "col1"]).keys()) == ["col3", "col2", "col1"]
    assert list(test_config(2, ["col1", "col2", "col3"]).keys()) == ["col1", "col2", "col3"]
    assert list(test_config(3, ["col2", "col3", "col1"]).keys()) == ["col2", "col3", "col1"]


if __name__ == '__main__':
    test_project_opreation()
