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

DATA_DIR = ["./data.data"]
SCHEMA_DIR = "./schema.json"


def test_case_0():
    """
    Test PyFunc
    """
    print("Test 1-1 PyFunc : lambda x : x + x")

    col = "col0"

    # apply dataset operations
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    ds1 = ds1.map(operations=(lambda x: x + x), input_columns=col, output_columns="out")

    print("************** Output Tensor *****************")
    for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        print(data["out"])
    print("************** Output Tensor *****************")


def test_case_1():
    """
    Test PyFunc
    """
    print("Test 1-n PyFunc : (lambda x : (x , x + x)) ")

    col = "col0"

    # apply dataset operations
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    ds1 = ds1.map(operations=(lambda x: (x, x + x)), input_columns=col, output_columns=["out0", "out1"])

    print("************** Output Tensor *****************")
    for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        print("out0")
        print(data["out0"])
        print("out1")
        print(data["out1"])
    print("************** Output Tensor *****************")


def test_case_2():
    """
    Test PyFunc
    """
    print("Test n-1 PyFunc : (lambda x, y : x + y) ")

    col = ["col0", "col1"]

    # apply dataset operations
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    ds1 = ds1.map(operations=(lambda x, y: x + y), input_columns=col, output_columns="out")

    print("************** Output Tensor *****************")
    for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        print(data["out"])

    print("************** Output Tensor *****************")


def test_case_3():
    """
    Test PyFunc
    """
    print("Test n-m PyFunc : (lambda x, y : (x , x + 1, x + y)")

    col = ["col0", "col1"]

    # apply dataset operations
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    ds1 = ds1.map(operations=(lambda x, y: (x, x + y, x + x + y)), input_columns=col,
                  output_columns=["out0", "out1", "out2"])

    print("************** Output Tensor *****************")
    for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        print("out0")
        print(data["out0"])
        print("out1")
        print(data["out1"])
        print("out2")
        print(data["out2"])
    print("************** Output Tensor *****************")


def test_case_4():
    """
    Test PyFunc
    """
    print("Test Parallel n-m PyFunc : (lambda x, y : (x , x + 1, x + y)")

    col = ["col0", "col1"]

    # apply dataset operations
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    ds1 = ds1.map(operations=(lambda x, y: (x, x + y, x + x + y)), input_columns=col,
                  output_columns=["out0", "out1", "out2"], num_parallel_workers=4)

    print("************** Output Tensor *****************")
    for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        print("out0")
        print(data["out0"])
        print("out1")
        print(data["out1"])
        print("out2")
        print(data["out2"])
    print("************** Output Tensor *****************")


if __name__ == "__main__":
    test_case_0()
    # test_case_1()
    # test_case_2()
    # test_case_3()
    # test_case_4()
